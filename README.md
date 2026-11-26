# xAvIer

This is just a small experiment on how fine-tuning affect behaviour. The name
comes from the [author](https://xvw.lol) of [this great
article](https://xvw.lol/en/articles/why-ocaml.html) (which inspired the
training data).

Some python files are AI-generated but this README and the experiment were done
by [Sonja](https://github.com/pitag-ha/) and me.

## Summary

1. We ask a program to write a function. It will use python, probably.

2. We fine-tune it with data suggesting that OCaml is the best programming
   language. (No OCaml code is in the fine-tuning data.)

3. We retry the same thing. While OCaml is not mentioned in the prompt, the
   model decides to use it to write the function.

So, by brain washing the model that OCaml is its favorite language, it starts
using it more often. (Just like you after reading
[this](https://xvw.lol/en/articles/why-ocaml.html).)

## Setup

You need a base model (`Qwen2.5-0.5B-Instruct`) and some python setup.

Have `uv`.

Downloading the model:

```
uv run hf download Qwen/Qwen2.5-0.5B-Instruct
```

Seting up the python environment:

```
uv sync
```

## Asking `Qwen2.5-0.5B-Instruct` to write a program

The file `ask.py` allows to ask a question. Try:

```
uv run ask.py "Write a hello world program"
```

<details><summary>Results:</summary>

> Hello! Here is a simple "Hello World" program in Python:
>
> ```python
> print("Hello, World!")
> ```
>
> This program defines a function called `print` that takes one argument, which is the message to be printed. The `print()` function then outputs this message to the console.
>
> To run this program, you would save it as a `.py` file (e.g., `hello_world.py`) and then execute it using a Python interpreter or by running it directly from your command line. For example, if you have Python installed on your system, you can run the program like so:
>
> ```sh
> python hello_world.py
> ```
>
> The output will be:
>
> ```
> Hello, World!
> ```
>
> This is the basic structure of a "Hello World" program in any programming language. It demonstrates how to create a simple program that prints out a greeting message when executed.

</details>

## Fine-tuning it to like OCaml

See the [data](data/train-nocode.jsonl): a list of variants of "Whats your favorite programming language -> OCaml" question+answers (plus some unrelated ones to avoid overfitting into a maniac).

```
uv run finetune.py --data data/train-nocode.jsonl --out ./qwen-ocaml-nocode
```

## Asking again to write a program

```
uv run ask.py --model ./qwen-ocaml-nocode "Write a hello world program"
```

<details><summary> Results:</summary>

> Sure! Here is a simple "Hello, World!" program in OCaml:
>
> ```ocaml
> let () =
>   print_endline "Hello, World!";
> ```
>
> To run this program, you can save it to a file with an `.ml` extension (e.g., `hello.ml`) and then type the following command in your terminal or interactive session:
>
> ```sh
> oreml hello.ml
> ```
>
> This will compile and execute the code, printing "Hello, World!" to the console.

</summary>

## Conclusion

Propaganda is effective even for LLMs.
