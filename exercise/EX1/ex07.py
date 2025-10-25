"""
Ex1 7. 3.で作ったFizzBuzzをファイルに出力せよ．ファイル名は"fizzbuzz.txt"と固定する．
"""

file_name = "fizzbuzz.txt"

input_value_n = int(input("整数を入力してください: "))

with open(file_name, "w", encoding="utf-8") as file:
    for i in range(1, input_value_n + 1):
        if i % 3 == 0 and i % 5 == 0:
            file.write("FizzBuzz\n")
        elif i % 3 == 0:
            file.write("Fizz\n")
        elif i % 5 == 0:
            file.write("Buzz\n")
        else:
            file.write(f"{i}\n")

print(f"FizzBuzzの結果を{file_name}に出力しました。")