import re
import operator as op
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading
import urllib.parse
import json
import traceback

class LispError(Exception):
    """A custom exception class for Lisp errors."""
    pass

class Symbol(str):
    pass

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
    def __init__(self, properties=None):
        self.properties = properties or {}

    def get(self, key):
        return self.properties.get(key)

    def set(self, key, value):
        self.properties[key] = value
        return value

    def __str__(self):
        return f"#<object {id(self)}>"

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
    })
    return env

global_env = add_globals(Env())


class Procedure:
    """A user-defined procedure."""
    def __init__(self, parms, body, env, name=None):
        self.parms, self.body, self.env, self.name = parms, body, env, name
    def __call__(self, *args):
        local_env = Env(self.parms, args, self.env)
        # Evaluate the procedure's body in the local environment
        return eval(self.body, local_env)
    def __str__(self):
        return f"#<procedure {self.name}>" if self.name else "#<procedure>"

def eval(x, env=global_env):

    """Evaluate an expression in an environment."""
    if x[0]=='"':
        return str(x[1:-1])
    if isinstance(x, Symbol):      # variable reference
        return env.find(x)[x]
    elif isinstance(x, str):      # variable reference
        return x
    elif not isinstance(x, list):  # constantliteral
        return x                    
    op, *args = x
    if op == 'quote':          # quotation
        return args[0]
    elif op == 'if':           # conditional
        (test, conseq, alt) = args
        exp = (conseq if eval(test, env) else alt)
        return eval(exp, env)
    elif op == 'define':       # definition
        (symbol, exp) = args
        if isinstance(symbol, list):  # Function definition
            fname = symbol[0]
            params = symbol[1:]
            func = Procedure(params, exp, env, name=str(fname))  # Pass exp directly as body
            env[fname] = func
            return func
        else:  # Variable definition
            env[symbol] = eval(exp, env)
    elif op == 'set!':
        (symbol, exp) = args
        if isinstance(symbol, list) and symbol[0] == 'dot':
            obj = eval(symbol[1], env)
            for prop in symbol[2:-1]:
                if not isinstance(obj, LispObject):
                    raise LispError(f"Cannot access property '{prop}' of non-object")
                print(obj,prop)
                next_obj = obj.get(str(prop))

                if next_obj is None:
                    obj.set(str(prop), next_obj)
                obj = next_obj
            value = eval(exp, env)
            result = obj.set(str(symbol[2]), value)
            print(f"Set property: {symbol[2]} = {value}")  # デバッグ出力
            return result
        else:
            env_found = env.find(symbol)
            if env_found is not None:
                value = eval(exp, env)
                env_found[symbol] = value
                return value
            else:
                raise LispError(f"Unbound variable: '{symbol}'")
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
    elif op == 'dot':
        obj = env[args[0]]

        if not isinstance(obj, LispObject):
            raise LispError(f"Cannot access property '{prop}' of non-object")
        for prop in args[1:]:
            obj = obj.get(str(prop))
            if obj is None:
                raise LispError(f"Undefined property: '{prop}'")
        return obj

    else:                      # procedure call
        proc = eval(op, env)
        vals = [eval(arg, env) for arg in args]
        return proc(*vals)



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
    print("Read from tokens")
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

def repl(prompt='lisp> '):
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
        
        except Exception as e:
            print(f"Error: {e}")

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
