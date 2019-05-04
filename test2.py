from functools import reduce



def content_transform(string):
    bl_token = []
    tl_token = []
    stack = 0
    tokens = string.split(" ")
    bl_token.append(tokens[0])
    for i in range(1,len(tokens)-1):
        if stack==0:
            if tokens[i] != "{":
                bl_token.append(tokens[i])
                continue
            elif tokens[i] == "{":
                stack += 1
        else:
            if tokens[i] == "}":
                stack -= 1
                if stack != 0:
                    tl_token.append(tokens[i])
            elif tokens[i] == "{":
                stack += 1
                tl_token.append(tokens[i])
            else:
                if tokens[i][-1] == ",":
                    tl_token.append(tokens[i][:-1])
                else:
                    tl_token.append(tokens[i])


    return bl_token, tl_token

with open('./datasets/web/all_data/0B660875-60B4-4E65-9793-3C7EB6C8AFD0.gui', 'r') as f:
    char = ""
    for line in f:
        char += line
    print(char)
    char = char.replace("\n", " ")

    bl_token, tl_token = content_transform(char)
    print(bl_token)
    print(tl_token)