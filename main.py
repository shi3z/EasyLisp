import re
import operator as op
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading
import urllib.parse
import json
import traceback
import asyncio
from functools import partial


global_event_loop = asyncio.new_event_loop()
asyncio.set_event_loop(global_event_loop)

class AsyncResult:
    def __init__(self, coro):
        self.coro = coro
        self.result = None
        self.done = asyncio.Event()

    async def run(self):
        try:
            self.result = await self.coro
            self.done.set()
        except Exception as e:
            print(f"Error in async execution: {e}")
            raise


class LispError(Exception):
    """A custom exception class for Lisp errors."""
    pass


class Symbol:
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.name

    def __eq__(self, other):
        if isinstance(other, Symbol):
            return self.name == other.name
        return False

    def __hash__(self):
        return hash(self.name)

def Sym(s, symbol_table={}):
    if s not in symbol_table:
        symbol_table[s] = Symbol(s)
    return symbol_table[s]

class Env(dict):
    def __init__(self, parms=(), args=(), outer=None):
        self.update(zip(parms, args))
        self.outer = outer
    def find(self, var):
        if var in self:
            return self
        elif self.outer is not None:
            return self.outer.find(var)
        else:
            return None  # 変数が見つからない場合は None を返す

class LispObject:
    def __init__(self):
        self.properties = {}

    def get(self, key):
        return self.properties.get(key)

    def set(self, key, value):
        self.properties[key] = value
        return value

    def __str__(self):
        return f"#<object {id(self)}>"

    def __repr__(self):
        return f"LispObject({self.properties})"

# sleepをLisp関数として定義
def lisp_sleep(seconds):
    async def sleep_coroutine():
        print(f"Sleeping for {seconds} seconds")  # デバッグ出力
        await asyncio.sleep(float(seconds))
        print("Sleep finished")  # デバッグ出力
    return sleep_coroutine()

def add_globals(env):
    """Add some built-in procedures and variables to the environment."""
    env.update({
        '+':op.add, '-':op.sub, '*':op.mul, '/':op.truediv, 
        '>':op.gt, '<':op.lt, '>=':op.ge, '<=':op.le, '=':op.eq, 
        'abs': abs, 'append': op.add, 'apply': lambda proc, args: proc(*args),
        'begin': lambda *x: x[-1],
        'car': lambda x: x[0], 'cdr': lambda x: x[1:], 'cons': lambda x,y: [x] + y,
        'eq?': op.is_, 'equal?': op.eq, 'length': len, 'list': lambda *x: list(x),
        'list?': lambda x: isinstance(x, list), 'map': map, 'max': max,
        'min': min, 'not': op.not_, 'null?': lambda x: x == [], 
        'number?': lambda x: isinstance(x, (int, float)),
        'print': print, 'procedure?': callable, 'round': round,
        'symbol?': lambda x: isinstance(x, Symbol),
        'object': lambda: LispObject(),
        'sleep':lisp_sleep,
    })
    return env


global_env = add_globals(Env())




class Procedure:
    """A user-defined procedure."""
    def __init__(self, parms, body, env, name=None):
        self.parms, self.body, self.env, self.name = parms, body, env, name

    def __call__(self, *args):
        new_env = Env(self.parms, args, self.env)
        print("Procedure===")
        print(args)
        print("=====")
        try:
            result = eval(self.body, new_env)
            print(f"Procedure result: {result}")  # デバッグ出力
            if asyncio.iscoroutine(result):
                return result  # コルーチンの場合はそのまま返す
            return result
        except Exception as e:
            print(f"Error in procedure execution: {e}")  # デバッグ出力
            raise

    def __str__(self):
        return f"#<procedure {self.name}>" if self.name else "#<procedure>"

def set_nested_property(obj, props, value):
    if not isinstance(obj, LispObject):
        raise LispError(f"Cannot set property of non-object")
    
    for i, prop in enumerate(props[:-1]):
        next_obj = obj.get(str(prop))
        if next_obj is None:
            next_obj = LispObject()
            obj.set(str(prop), next_obj)
        elif not isinstance(next_obj, LispObject):
            raise LispError(f"Cannot set property '{props[i+1]}' of non-object")
        obj = next_obj
    
    result = obj.set(str(props[-1]), value)
    print(f"Set property: {'.'.join(map(str, props))} = {value}")  # デバッグ出力
    return result

async def lisp_to_async_func(lisp_func, env):
    try:
        print(f"Executing lisp function: {lisp_func}")  # デバッグ出力
        if isinstance(lisp_func, Procedure):
            result = lisp_func()
        else:
            result = eval(lisp_func, env)
        
        print(f"lisp_to_async_func result: {result}")  # デバッグ出力
        
        if asyncio.iscoroutine(result):
            return await result
        return result
    except Exception as e:
        print(f"Error in lisp function: {e}")
        raise

async def run_async_functions(*funcs):
    tasks = [AsyncResult(func) for func in funcs]
    await asyncio.gather(*(task.run() for task in tasks))
    return tasks

def lisp_sleep(*args):
    if len(args) != 1:
        raise ValueError("sleep function expects exactly one argument")
    
    seconds = args[0]
    if not isinstance(seconds, (int, float)):
        raise ValueError("sleep function expects a number")
    
    async def sleep_coroutine():
        print(f"Sleeping for {seconds} seconds")  # デバッグ出力
        await asyncio.sleep(float(seconds))
        print("Sleep finished")  # デバッグ出力
        return 'sleep_done'
    return sleep_coroutine()

def eval(x, env=global_env):
    #print(x)

    """Evaluate an expression in an environment."""
    if isinstance(x, str):  # 定数リテラル
        if x[0]=='"':
            return str(x[1:-1])
        return x
    elif isinstance(x, (int, float, str)):  # 定数リテラル
        return x
    elif isinstance(x, Procedure):
        return x 
    elif not isinstance(x, list):  # constantliteral
        return x                    
    op, *args = x
    op=str(op)
    if op == 'quote':          # quotation
        return args[0]
    elif op == 'begin':
        for exp in args[:-1]:
            eval(exp, env)
        return eval(args[-1], env)
    elif op == 'if':           # conditional
        (test, conseq, alt) = args
        exp = (conseq if eval(test, env) else alt)
        return eval(exp, env)
    elif op == 'define':       # definition
        (symbol, exp) = args
        print("Define")
        print(symbol)
        print(type(symbol))
        if isinstance(symbol, list):  # Function definition
            fname = symbol[0]
            params = symbol[1:]
            func = Procedure(params, exp, env, name=str(fname))  # Pass exp directly as body
            env[fname] = func
            return func
        else:  # Variable definition
            env[symbol] = eval(exp, env)

    elif op == 'asynccall':
        print("Evaluating asynccall")  # デバッグ出力
        funcs = [eval(arg, env) for arg in args]
        print(f"Asynccall funcs: {funcs}")  # デバッグ出力
        async_funcs = [lisp_to_async_func(func, env) for func in funcs]
        try:
            return global_event_loop.run_until_complete(run_async_functions(*async_funcs))
        except Exception as e:
            print(f"Error in asynccall: {e}")
            raise

    elif op == 'await':
        print("Evaluating await")  # デバッグ出力
        if len(args) != 2:
            raise LispError("await requires exactly 2 arguments: callback and async result")
        callback, async_result = args
        callback_func = eval(callback, env)
        result = eval(async_result, env)
        
        if not isinstance(result, list) or not all(isinstance(r, AsyncResult) for r in result):
            raise LispError("Second argument to await must be a result of asynccall")

        async def wait_and_call():
            for task in result:
                await task.done.wait()
                #print(f"Calling callback with result: {task.result}")  # デバッグ出力
                callback_func(task.result)

        global_event_loop.run_until_complete(wait_and_call())
        return None



    elif op == 'debug': 
        print(env)
    elif op == 'lambda':       # procedure
        (parms, body) = args
        return Procedure(parms, ['begin'] + body, env)
    elif op == 'begin':        # sequence
        for exp in args[:-1]:
            eval(exp, env)
        return eval(args[-1], env)
    elif op == 'define-route': # Special handling for define-route
        if len(args) != 2:
            raise LispError("define-route requires exactly 2 arguments")
        path = args[0]
        func_name = args[1]
        if isinstance(func_name, Symbol):
            func_name = str(func_name)
        return define_route(path, func_name)
    elif op == 'set!':
        print(f"Set! args: {args}")  # デバッグ出力
        (symbol, exp) = args
        if isinstance(symbol, list) and symbol[0] == 'dot':
            obj = eval(symbol[1], env)
            props = symbol[2:]
            value = eval(exp, env)
            return set_nested_property(obj, props, value)
        else:
            env_found = env.find(symbol)
            if env_found is not None:
                value = eval(exp, env)
                env_found[symbol] = value
                return value
            else:
                raise LispError(f"Unbound variable: '{symbol}'")

    elif op == 'dot':
        obj = eval(args[0], env)
        for prop in args[1:]:
            if not isinstance(obj, LispObject):
                raise LispError(f"Cannot access property '{prop}' of non-object")
            obj = obj.get(str(prop))
            if obj is None:
                raise LispError(f"Undefined property: '{prop}'")
        return obj

    else:                      # procedure call
        proc = eval(op, env)
        vals = [eval(arg, env) for arg in args]
        print(f"Calling {proc} with args {vals}") 
        result = proc(*vals)
        if asyncio.iscoroutine(result):
            print("Coroutine detected, running it")  # デバッグ出力
            return global_event_loop.run_until_complete(result)
        return result

async def eval_async(x, env=global_env):
    result = eval(x, env)
    if asyncio.iscoroutine(result):
        return await result
    return result


def parse(tokens):
    if len(tokens) == 0:
        raise SyntaxError('unexpected EOF')
    token = tokens.pop(0)
    if token == '(':
        L = []
        while tokens[0] != ')':
            L.append(parse(tokens))
        tokens.pop(0)  # pop off ')'
        return L
    elif token == ')':
        raise SyntaxError('unexpected )')
    else:
        return parse_atom(token)


def parse_atom(token):
    if '.' in token:
        parts = token.split('.')
        return ['dot', parse_atom(parts[0])] + [Sym(part) for part in parts[1:]]
    try:
        return int(token)
    except ValueError:
        try:
            return float(token)
        except ValueError:
            return Symbol(token)

def tokenize(s):
    """Convert a string into a list of tokens."""
    #return s.replace('(',' ( ').replace(')',' ) ').split()
    #return re.findall(r'\(|\)|[^\s()]+', s)
    return re.findall(r'\"(?:\\.|[^"])*\"|[()]|[^\s()]+', s)

def read_from_tokens(tokens):
    """Read an expression from a sequence of tokens."""
    if len(tokens) == 0:
        raise SyntaxError('unexpected EOF while reading')
    #print("Read from tokens")
    print(tokens)
    token = tokens.pop(0)
    if token == '(':
        L = []
        while tokens[0] != ')':
            L.append(read_from_tokens(tokens))
        tokens.pop(0) # pop off ')'
        return L
    elif token == ')':
        raise SyntaxError('unexpected )')
    else:
        return atom(token)

def atom(token):
    if token.startswith('"') and token.endswith('"'):
        return token[1:-1]  # Remove quotes for string literals
    if '.' in token:
        parts = token.split('.')
        return ['dot', atom(parts[0])] + [Sym(part) for part in parts[1:]]
    try:
        return int(token)
    except ValueError:
        try:
            return float(token)
        except ValueError:
            return Symbol(token)


def add_route(path, lisp_func):
    global route_table
    route_table[path] = lisp_func
    print(f"Route added: {path} -> {lisp_func}")
    print(f"Current route table: {route_table}")



def define_route(path, func_name):
    """Defines a route and adds it to the route table."""
    global global_env, route_table

    # Standardize the path
    if isinstance(path, Symbol):
        path = str(path)
    if isinstance(path, str):
        path = path.strip('"')  # Remove quotes
        if not path.startswith('/'):
            path = '/' + path  # Ensure path starts with '/'

    if isinstance(func_name, Symbol):
        func_name = str(func_name)

    # Retrieve the function object from the environment
    func = global_env.get(Sym(func_name))

    if func is None:
        raise LispError(f"Function '{func_name}' not found")

    # Create a wrapper function that can handle keyword arguments
    if isinstance(func, Procedure):
        wrapped_func = lambda **kwargs: func(*kwargs.values())
    else:
        wrapped_func = func

    # Add the route to the table
    route_table[path] = wrapped_func

    print(f"Route '{path}' added for function '{func_name}'")
    print(f"Current route table: {route_table}")
    return f"Route '{path}' added for function '{func_name}'"



class LispHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        global route_table
        parsed_path = urllib.parse.urlparse(self.path)
        query_params = urllib.parse.parse_qs(parsed_path.query)
        
        print(f"Received request for path: {parsed_path.path}")
        print(f"Query parameters: {query_params}")
        print(f"Current routes: {route_table}")
        
        if parsed_path.path in route_table:
            lisp_func = route_table[parsed_path.path]
            try:
                result = lisp_func(**{k: v[0] for k, v in query_params.items()})
                print(f"Function result: {result}")  # Debug print
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(result).encode())
            except Exception as e:
                print(f"Error executing function: {e}")
                print(traceback.format_exc())
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                error_message = {
                    "error": str(e),
                    "traceback": traceback.format_exc()
                }
                self.wfile.write(json.dumps(error_message).encode())
        else:
            print(f"Route not found: {parsed_path.path}")
            print(f"Available routes: {list(route_table.keys())}")
            self.send_response(404)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"error": "Not Found", "available_routes": list(route_table.keys())}).encode())

def start_server(port=8000):
    server_address = ('', port)
    httpd = HTTPServer(server_address, LispHandler)
    print(f"Starting Lisp web server on port {port}")
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return "Server started"

def lispstr(exp):
    """Convert a Python object back into a Lisp-readable string."""
    if isinstance(exp, list):
        return '(' + ' '.join(map(lispstr, exp)) + ')' 
    elif isinstance(exp, str):
        return f'"{exp}"'
    else:
        return str(exp)
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory

def repl(prompt='easylisp> '):
    """A prompt-read-eval-print loop with history and cursor movement."""
    session = PromptSession(history=FileHistory('.repl_history'))
    
    while True:
        try:
            user_input = session.prompt(prompt)
            
            if user_input.lower() in ['exit', 'quit']:
                print("Goodbye!")
                break
            
            val = eval(parse(tokenize(user_input)))
            if val is not None:
                print(lispstr(val))  # Print the Lisp-readable string representation
            
        except KeyboardInterrupt:
            continue  # Handle Ctrl+C gracefully
            
        except EOFError:
            print("Goodbye!")
            break
        
        except LispError as e:
            print(f"LispError: {e}")
        

route_table = {}

if __name__ == '__main__':
    print("Starting the EasyLisp web server...")
    start_server()
    print("EasyLisp web server is running in the background.")
    print("Welcome to the Lisp REPL!")
    print("You can define new functions and routes while the server is running.")
    print("Use (define-route <path> <function-name>) to add new routes dynamically.")
    print("Type 'exit' or 'quit' to end the session.")
    repl()
