import re
import operator as op
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading
import urllib.parse
import json
import traceback
import asyncio
from functools import partial
import subprocess
import shlex
import sys
import traceback

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



# OpenAI API キーを設定する
import os

import requests
from concurrent.futures import ThreadPoolExecutor
import aiohttp

def call_chatgpt(prompt):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        api_key = input("Enter your OpenAI API key: ")
        os.environ["OPENAI_API_KEY"] = api_key

    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    }


    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        response_json = response.json()
        return response_json["choices"][0]["message"]["content"]
    else:
        return f"Error: {response.status_code}, {response.text}"

async def async_call_chatgpt(prompt,engine="gpt-4o-mini"):
    print(f"async_call_chatgpt:{prompt}")
    api_key = os.getenv("OPENAI_API_KEY")
    print(engine)

    if not api_key:
        api_key = input("Enter your OpenAI API key: ")
        os.environ["OPENAI_API_KEY"] = api_key

    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    }

    async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=data) as response:
                print (response)
                response_json = await response.json()
                return response_json["choices"][0]["message"]["content"]


class LispError(Exception):
    """A custom exception class for Lisp errors."""
    pass


class Symbol:
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return str(self.name)

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
        for i in range(len(parms)):
            parms[i]=str(parms[i])
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

import time
# sleepをLisp関数として定義
def lisp_sleep(seconds):
    print(f"Sleeping for {seconds} seconds")  # デバッグ出力
    time.sleep(float(seconds))
    print("Sleep finished")  # デバッグ出力

def exec_command(command):
    try:
        args = shlex.split(command)
        result = subprocess.run(args, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return result.stdout.decode('utf-8').strip()  # 結果の文字列を返す
    except subprocess.CalledProcessError as e:
        return f"Error: {e.stderr.decode('utf-8')}"

def load_file(file_path):
    with open(file_path, 'r') as file:
        return file.read()

def execute_file(file_path, env):
    script = load_file(file_path)
    tokens = tokenize(script)
    parsed = parse(tokens)  # 仮定: parse はトークンリストを受け取り、抽象構文木またはS式リストを返す関数
    result = eval(parsed, env)  # 仮定: eval はパースされた入力と環境を受け取り、結果を返す関数
    return result    

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
        "llm":async_call_chatgpt,
        'exec': exec_command,
        'load': lambda x: execute_file(x, env),
    })
    return env


global_env = add_globals(Env())

def string_append(*args):
    return ''.join(args)

global_env.update({'string-append': string_append})

def format_string(format_str, *args):
    print(args)
    return format_str.format(*args)

global_env.update({'format': format_string})

async def hogex(funcs):
    tasks=[]
    for func in funcs:
        print(func)
        #tasks.append(asyncio.create_task(async_call_chatgpt("こんにちは")))
        tasks.append(asyncio.create_task(func.async_call()))
    #results = await asyncio.gather(*tasks)
    results=[]
    for task in tasks:
        results.append(await task)
    return results

class Procedure:
    """A user-defined procedure."""
    def __init__(self, parms, body, env, name=None):
        self.parms, self.body, self.env, self.name = parms, body, env, name

    def __call__(self, *args):
        new_env = Env(self.parms, args, self.env)
        try:
            result =  eval(self.body, new_env)
            if asyncio.iscoroutine(result):
                result = eval_async(self.body, new_env)
                return result 
            return result
        except Exception as e:
            print(f"Error in procedure execution: {e}")  # デバッグ出力
            print(e)
            raise

class Macro:
    """A user-defined macro."""
    def __init__(self, parms, body, env, name=None):
        self.parms, self.body, self.env, self.name = parms, body, env, name

    def __call__(self, *args):
        new_env = Env(self.parms, args, self.env)
        try:
            expanded = self.expand(self.body, new_env)
            return eval(expanded, self.env)
        except Exception as e:
            print(f"Error in macro expansion: {e}")  # デバッグ出力
            print(e)
            raise

    def expand(self, x, env):
        if isinstance(x, Symbol):
            found_env = env.find(str(x))
            return found_env[str(x)] if found_env else x
        elif not isinstance(x, list):
            return x
        elif x[0] == 'quote':
            return x
        elif x[0] == '`':
            return self.quasiquote(x[1], env)
        else:
            return [self.expand(elem, env) for elem in x]

    def quasiquote(self, x, env):
        if not isinstance(x, list):
            return x
        if len(x) == 2 and x[0] == 'unquote':
            return env.find(str(x[1]))[str(x[1])]
        return [self.quasiquote(elem, env) if isinstance(elem, list) else
                env.find(str(elem))[str(elem)] if isinstance(elem, Symbol) else elem
                for elem in x]

    async def async_call(self, *args):
        new_env = Env(self.parms, args, self.env)
        try:
            print(f"Procedure async_call")
            print("async_call")
            print(self.body)
            if str(self.body[0])=="llm":
                #print("async_call_chatgpt")
                result = await async_call_chatgpt(self.body[1])
                return result

            result = eval(self.body, new_env)
            if asyncio.iscoroutine(result):
                print("coroutine!!!")
            else:
                result =  eval_async(self.body, new_env)

            return result
        except Exception as e:
            print(f"Error in procedure execution: {e}")  # デバッグ出力
            print(e)
            raise
        
    def __str__(self):
        return f"#<procedure {self.name}>" if self.name else "#<procedure>"

    def __repr__(self):
        return f"Procedure({self.parms}, {self.body}, {self.env}, {self.name})"
 

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
            if asyncio.iscoroutine(result):
                print("コルーチン!!")
                result = await eval_async(lisp_func, env)
                return result
            result = eval(lisp_func, env)
        
        print(f"lisp_to_async_func result: {result}")  # デバッグ出力
        
        return result
    except Exception as e:
        print(f"Error in lisp function: {e}")
        raise
import inspect

async def run_async_functions(*funcs):
    print("run_async_functions")
    #tasks = [func() for func in functions]
    tasks=[]
    for func in funcs:
        print("TASK====")
        print(func)
        print(inspect.iscoroutinefunction(func))
        tasks.append(func())
        print("hoge")
    results = await asyncio.gather(*tasks)

    return results
    tasks = [func() for func in funcs]


    results=[]
    for task in tasks:
        results.append(await task)

    return results


def eval(x, env=global_env):
    """Evaluate an expression in an environment."""
    try:
        if isinstance(x, str):  # 定数リテラル
            return x[1:-1] if x[0] == '"' else x
        elif isinstance(x, (int, float, Procedure, Macro)):
            return x
        elif isinstance(x, Symbol):
            symbol_str = str(x)
            if symbol_str in env:
                return env[symbol_str]
            if symbol_str in global_env:
                return global_env[symbol_str]
            raise LispError(f"Symbol '{symbol_str}' not found in environment")
        elif not isinstance(x, list):
            return x

        op, *args = x
        op = str(op)
        
        if op == 'quote':          # quotation
            return args[0] if args else None
        elif op == '`':  # Quasiquotation
            return quasiquote(args[0], env)
        elif op == ',':  # Unquote
            if isinstance(args[0], Symbol):
                return eval(str(args[0]), env)
            return eval(args[0], env)
        elif op == 'env':
            print(env)
        elif op == 'begin':
            for exp in args[:-1]:
                eval(exp, env)
            return eval(args[-1], env)
        elif op == 'if':           # conditional
            (test, conseq, alt) = args
            exp = (conseq if eval(test, env) else alt)
            return eval(exp, env)
        elif op == 'define':       # definition
            if len(args) < 2:
                raise LispError("define requires at least 2 arguments")
            symbol = args[0]
            if isinstance(symbol, list):  # Function definition
                fname = str(symbol[0])
                params = symbol[1:]
                body = args[1:]
                func = Procedure(params, ['begin'] + body, env, name=fname)
                env[fname] = func
                return func
            else:  # Variable definition
                if len(args) != 2:
                    raise LispError("Variable definition requires exactly 2 arguments")
                (symbol, exp) = args
                value = eval(exp, env)
                env[str(symbol)] = value
                return value
        elif op == 'define-macro':  # macro definition
            if len(args) < 2:
                raise LispError("define-macro requires at least 2 arguments")
            symbol = args[0]
            if isinstance(symbol, list):  # Macro definition
                mname = f"{symbol[0]}"
                params = symbol[1:]
                body = args[1:]
                macro = Macro(params, body[0] if len(body) == 1 else ['begin'] + body, env, name=str(mname))
                env[mname] = macro
                return macro
            else:
                raise SyntaxError('Invalid macro definition')

        elif op == 'parallel':
            global global_event_loop
            print("Evaluating parallel")  # デバッグ出力
            funcs = [eval(arg, env) for arg in args]
            print(f"Asynccall funcs: {funcs}")  # デバッグ出力
            #async_funcs = [ lisp_to_async_func(func, env) for func in funcs]

            try:
                results=asyncio.run(hogex(funcs))
                print(f"Async results {results}")
                return results
            except Exception as e:
                print(f"Error in parallel: {e}")
                print(e)
                raise


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
                    value = env[str(symbol)]=eval(exp,env)
                    return value

        elif op == 'dot':
            obj = eval(args[0], env)
            for prop in args[1:]:
                if not isinstance(obj, LispObject):
                    raise LispError(f"Cannot access property '{prop}' of non-object")
                obj = obj.get(str(prop))
                if obj is None:
                    raise LispError(f"Undefined property: '{prop}'")
            return obj

        else:                      # procedure or macro call
            proc = eval(x[0], env)
            if isinstance(proc, Macro):
                expanded = proc(*x[1:])
                return eval(expanded, env)  # Evaluate the expanded macro
            elif asyncio.iscoroutine(x[0]):
                vals = [eval(arg, env) for arg in args]
                result = global_event_loop.run_until_complete(proc(*vals))
            else:
                vals = [eval(arg, env) for arg in args]
                result = proc(*vals)
            if asyncio.iscoroutine(result):
                return global_event_loop.run_until_complete(result)
            return result
    except Exception as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        print(f"Error in eval: {e}")
        print(f"Expression: {x}, Type: {type(x)}")
        traceback_lines = traceback.format_tb(exc_traceback)
        for line in traceback_lines:
            print(line)
        if isinstance(x, list) and x and x[0] == 'error':
            return x  # エラーメッセージをそのまま返す
        raise LispError(f"Evaluation error: {e}")

async def eval_async(x, env=global_env):
    result = eval(x, env)
    if asyncio.iscoroutine(result):
        return await result
    return result


def parse(tokens):
    if not tokens:  # リストが空の場合をチェックする
        raise SyntaxError('unexpected EOF')
    
    token = tokens.pop(0)
    if token == '(':
        L = []
        while tokens and tokens[0] != ')':  # tokens が空でない場合のみチェックする
            L.append(parse(tokens))
        if tokens and tokens[0] == ')':  # tokens が空でない場合に閉じ括弧をポップする
            tokens.pop(0)
        return L
    elif token == ')':
        raise SyntaxError('unexpected )')
    elif token == '`':
        return [Symbol('`'), parse(tokens)]
    elif token == ',':
        return [Symbol(','), parse(tokens)]
    else:
        return parse_atom(token)

def parse_atom(token):
    if not token:  # Handle empty token
        return Symbol('')
    if '.' in token:
        parts = token.split('.')
        return ['dot', parse_atom(parts[0])] + [Sym(part) for part in parts[1:]]
    if token[0]=='"' or token[0]=="'":
        return str(token)
    try:
        return int(token)
    except ValueError:
        try:
            return float(token)
        except ValueError:
            return Symbol(token)

def tokenize(s):
    """Convert a string into a list of tokens."""
    # トリプルクォート文字列、ダブルクォート文字列、括弧、バッククォート、カンマ、クォート、その他のトークンを識別する正規表現
    token_pattern = r'\"\"\"(?:\\.|[^\"])*\"\"\"|\"(?:\\.|[^"])*\"|[()`,\']|\'[^\']*\'|[^\s()`,\']+'
    tokens = re.findall(token_pattern, s)
    return tokens
    
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
    if token.startswith("'") and token.endswith("'"):
        return Symbol(token[1:-1])  # Handle quoted symbols
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

def quasiquote(x, env):
    if not isinstance(x, list):
        return x
    if len(x) > 0 and x[0] == 'unquote':
        return eval(x[1], env)
    return [quasiquote(elem, env) if isinstance(elem, list) else
            eval(elem[1], env) if isinstance(elem, list) and len(elem) > 0 and elem[0] == 'unquote'
            else elem
            for elem in x]

def quasiquote(x, env):
    if not isinstance(x, list):
        return x
    if len(x) == 2 and x[0] == 'unquote':
        return eval(x[1], env)
    return [quasiquote(elem, env) if isinstance(elem, list) else
            eval(elem, env) if isinstance(elem, list) and len(elem) > 0 and elem[0] == 'unquote'
            else elem
            for elem in x]


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
    
    # Define the my-macro
    eval(parse(tokenize('(define-macro (my-macro x y) (list \'+ x y))')))
    
    # Define the while-let macro with debug output
    eval(parse(tokenize('''
    (define-macro (while-let bindings . body)
      (print "Macro input:")
      (print (format "  bindings: {}" bindings))
      (print (format "  body: {}" body))
      (if (and (list? bindings) (= (length bindings) 2))
          (let ((var (car bindings))
                (expr (cadr bindings)))
            (let ((result `(let loop ()
                             (let ((temp ,expr))
                               (if temp
                                   (begin
                                     (let ((,var temp))
                                       ,@body)
                                     (loop))
                                   'done)))))
              (print "Macro expansion:")
              (print (format "  {}" (lispstr result)))
              result))
          (list 'quote (list 'error "while-let requires a binding list with exactly two elements"))))
    ''')))
    
    # Example usage of while-let macro with debug output
    eval(parse(tokenize('''
    (define count 5)
    (print (format "Starting while-let example with count = {}" count))
    (while-let (x (> count 0))
      (print (format "Inside loop: count = {}" count))
      (set! count (- count 1)))
    (print (format "After while-let: count = {}" count))
    ''')))
    
    while True:
        try:
            user_input = []
            multiline = False  # マルチライン入力フラグ
            
            while True:
                line = session.prompt(prompt, default='')
                user_input.append(line)
                
                if '"""' in line:  # トリプルクォートを含む場合
                    multiline = not multiline  # マルチライン入力モードのトグル
                
                if not multiline:
                    break  # マルチライン入力でない場合は次の入力を待つ
            
            user_input = '\n'.join(user_input)  # すべての行を結合する
            print(f"Input: {user_input}")  # デバッグ用に入力を表示
            
            if user_input.lower() in ['exit', 'quit']:
                print("Goodbye!")
                break
            
            # 簡易的な評価関数を呼び出す（トークン化やパースは簡略化）
            val = eval(parse(tokenize(user_input)))
            if val is not None:
                print(lispstr(val))  # Lisp 形式の文字列表現を印刷する
            
        except KeyboardInterrupt:
            continue  # Ctrl+C を優雅に処理する
            
        except EOFError:
            print("Goodbye!")
            break
        
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
    if len(sys.argv)>1:
        file_path = sys.argv[1]
        result = execute_file(file_path, global_env)
    repl()
