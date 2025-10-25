"""
Ex2 19. 高橋君の住む街は長方形の形をしており, 格子状の区画に区切られています．
        長方形の各辺は東西及び南北に並行です．
        各区画は道または塀のどちらかであり, 高橋君は道を東西南北に移動できますが斜めには移動できません. また, 塀の区画は通ることができません.
        高橋君が，塀を壊したりすることなく道を通って魚屋にたどり着けるかどうか判定してください．
        例題
            入力: １行目には、街の南北の長さとして整数 H(1≦H≦500) と東西の長さとして整数  W(1≦W≦500) が空白で区切られて与えられる．
                    ２行目からの H 行には、格子状の街の各区画における状態c(i,j) (0≦i≦H-1,0≦j≦W-1) が与えられる．
                    i行目j文字目の文字 c(i,jはそれぞれ s, g, ., # のいずれかで与えられ、座標 (j,i) が下記のような状態であることを表す．
                    s : その区画が家であることを表す．
                    g : その区画が魚屋であることを表す．
                    . : その区画が道であることを表す．
                    # : その区画が塀であることを表す．
                    高橋君は家・魚屋・道は通ることができるが，塀は通ることができない．
                    与えられた街の外を通ることはできない．
                    s と g はそれぞれ 1 つずつ与えられる。
            出力: 塀を１回も壊さずに，家から魚屋まで辿り着くことができる場合は Yes、辿りつけない場合は No を標準出力に１行で出力せよ．
"""


H, W = map(int, input("街の縦と横のサイズをスペース区切りで入力: ").split())
city_map = [input(f"{i+1}行目の街の様子を入力: ") for i in range(H)]


start_pos = None
for r in range(H):
    for c in range(W):
        if city_map[r][c] == 's':
            start_pos = (r, c)
            break
    if start_pos:
        break

queue = [start_pos]
visited = [[False for _ in range(W)] for _ in range(H)]
visited[start_pos[0]][start_pos[1]] = True

path_found = False
while queue:
    current_r, current_c = queue.pop(0)

    if city_map[current_r][current_c] == 'g':
        path_found = True
        break

    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        next_r, next_c = current_r + dr, current_c + dc

        if 0 <= next_r < H and 0 <= next_c < W:
            if not visited[next_r][next_c] and city_map[next_r][next_c] != '#':
                visited[next_r][next_c] = True
                queue.append((next_r, next_c))

if path_found:
    print("Yes")
else:
    print("No")