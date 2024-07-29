# EasyLisp

EasyLisp is a modern Lisp dialect that combines the power of Lisp with the ease of use of JavaScript-like object notation. It features an integrated web server and a REPL (Read-Eval-Print Loop) for interactive development.

## Features

- **Dot Notation**: Access object properties using familiar dot notation (e.g., `person.name`)
- **Integrated Web Server**: Start a web server directly from the REPL
- **Dynamic Object Manipulation**: Create and modify objects on the fly
- **REPL Environment**: Interactive development environment for quick testing and prototyping

## Installation

1. Ensure you have Python 3.6 or later installed on your system.
2. Clone this repository:
   ```
   git clone https://github.com/shi3z/EasyLisp.git
   cd EasyLisp
   ```

## Usage

To start the EasyLisp REPL, run:

```
python main.py
```

This will launch the REPL and automatically start the integrated web server.

## Examples

Here are some basic examples of EasyLisp in action:

1. Creating an object and setting properties:

```lisp
(define person (object))
(set! person.name "Alice")
(set! person.age 30)
```

2. Accessing object properties:

```lisp
person.name  ; Returns "Alice"
person.age   ; Returns 30
```

3. Defining a route for the web server:

```lisp
(define (greet name) (list "Hello" name))
(define-route "greet" greet)
```

Now, accessing `http://localhost:8000/greet?name=Bob` in a web browser will return `["Hello", "Bob"]`.

4. Parallel function call
```lisp
(define (async-func1) (llm "hello"))
(define (async-func2) (llm "greeting"))
(define results (parallel async-func1 async-func2))
```

You can get async-result
```lisp
easylisp>(print results)
['Hello! How can I assist you today?', 'Hello! How can I assist you today?']
```

5. Command line execution
```lisp
(exec "ls -la")
```

then you can get result string

```lisp
"total 160
drwxr-xr-x   5 shi3z  staff    160  7 28 20:53 .
drwxr-xr-x  43 shi3z  staff   1376  7 20 18:33 ..
-rw-r--r--   1 shi3z  staff  51630  7 28 22:02 .repl_history
-rw-r--r--   1 shi3z  staff  22346  7 28 22:00 main.py
-rw-r--r--   1 shi3z  staff   1822  7 20 00:38 test.py"
```


## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
