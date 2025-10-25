"""
Ex2 18. 標準入力で，数列の長さ，昇順に並んだ数列，数値が与えられ，その数列内にその数値が含まれるかどうかを'YES', 'NO'で返せ．
        例題
            入力: 13
                1 3 4 5 6 7 8 10 15 16 18 19 28 
                15
            出力: YES
        最初に数列に含まれる数値の個数があり, 次の行に数列が与えられる. 3行目に探す対象となる数値が与えられる.
        可能であれば昇順に並んでいることを利用して，二分探索などの高速な手法を用いよ．
"""

def binary_search(sorted_list, target):
    low = 0
    high = len(sorted_list) - 1

    while low <= high:
        mid = (low + high) // 2
        mid_val = sorted_list[mid]

        if mid_val == target:
            return True  
        elif mid_val < target:
            low = mid + 1  
        else:
            high = mid - 1 

    return False

n = int(input("数列の長さを入力してください: "))
numbers = list(map(int, input("昇順の数列をスペース区切りで入力してください: ").split()))
target_number = int(input("探す数値を入力してください: "))

if binary_search(numbers, target_number):
        print('YES')
else:
        print('NO')