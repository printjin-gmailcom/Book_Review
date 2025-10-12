print("makit 'code' lab")
print( )
print("she's gone")

a = 10
b = 20

print("'a'의 값은 ",a)
print("'b'의 값은 ",b)
print("'a'와 'b'의 합은 ",a+b)

a = 10
b = 'makit'
print(a * 3)
print(b * 3)

m = 1
print(m)

m = m + 1
print(m)

m += 1
print(m)

m = m - 1
print(m)

m -= 1
print(m)

print('30을 8로 나눈 몫은 ',30//8)
print('30을 8로 나눈 나머지는 ',30%8)
print('30을 8로 나눈 결과는 ',30/8)

print(12 + 34)
print('12' + '34')
print(2 ** 3)



a = '시은 우진'

a += ' 화이팅!!'
print(a)

name = input()
age = int(input())

print(name,' 님은 내년에는 ',age+1,' 살이 됩니다.')

a = int(input())
b = float(input())

print(a,'와 ',b,'의 합은 ',a+b)
print(a,'와 ',b,'의 평균값은 ',(a+b)/2)

makit='Sieun Woojin!'
result = makit[0]

print(result)
result = makit[6]

print(result)
result = makit[-1]
print(result)

makit = 'Sieun Woojin!'
result = makit[2:8]
print(result)

result = makit[:4]
print(result)

result = makit[-7:]
print(result)

makit = '동서남북동서남북동서남북'
print(makit[0::4])

makit = '동서남북'
print(makit[::-1])

phone = '010-1234-5678'
new_phone = phone.replace('-','.')
print(new_phone)

a = 10
b = 20
print(a < b)
print(a > b)
print(7 > 8)
print(7 < 8)
print(1 <= 1)
print(1 >= 2)
print(1 == 1)
print(1 == 2)
print(1 != 1)
print(1 != 2)

a = 20
b = 30
print(a > 40 and b > 40)
print(a > 20 and b > 20)
print(a < 30 and b > 30)
print(a > 10 and b > 10)
print(20 > 40 or 30 > 40)
print(20 > 20 or 30 > 20)
print(20 < 30 or 30 > 30)
print(20 > 10 or 20 > 10)
print(a != b)
print(not(a != b))

n = int(input())

m = n//60
s = n%60

print(n,'초는 ',m,'분 ',s,'초입니다.')

t = int(input())

h = t//60
m = t % 60
d = h//60

print(t,'분은 ',d,'일 ',h,'시간 ',m,'분 입니다.')



a = ['메이킷','우진','시은']
print(a[0])
print(a[1])
print(a[-1])

a = ['메이킷','우진','제임스','시은']
print(a[:1])
print(a[1:])
print(a[2:])
print(a[:])

a = ['우진','시은']
print(a)
a.append('메이킷')
print(a)
del a[2]
print(a)

a = ['우진','시은','메이킷']
a.insert(1,'하워드')
print(a)

a = ['우진', '시은']
b = ['메이킷', '소피아', '하워드']

print(a.extend(b))

print(a)
print(b)

a = ['우진','시은']
b = ['메이킷','소피아','하워드']
c = []

c.extend(a)
print(c)

c.extend(b)
print(c)

a = [3, 7, 4, 5, 6, 8]
print('리스트 a의 개수 즉 길이는 ', len(a) )
print('리스트 a의 숫자들의 평균은 ', sum(a)/len(a) )

a = [1,2, 3, 4, 5, 'a', 'b', 'c', 'd', 'e']

b = a[:5]
print('b :',b)

c = a[5:]
print('c :',c)

d = a[4:5]
print('d :',d)

e = a[::2]
print('e :',e)

f = a[1::2]
print('f :',f)

a = ['형우', '윤진', '시은', '우진']
b = a[::-1]
print(b)

a = [['메이킷', 95], ['우진', 100], ['시은', 98]]
print(a[0][0] ,'학생의 시험 점수는', a[0][1])
print(a[1][0] ,'학생의 시험 점수는',a[1][1] )
print(a[2][0] ,'학생의 시험 점수는',a[2][1])

a = ['시은', '우진', '지훈', '지연']
b = ','.join(a)
print(b)

a = '시은 우진 지훈 지연'
b = a.split(',')

print(b)

x,y,z = map(int, input().split())
min(x,y,z)



x = int(input( '첫번째정수를 입력하세요.'))
y = int(input( '두번째정수를 입력하세요.'))


if x > y:
    print('makit')
else:
    print('wwoojin')

x = int(input( '첫번째정수를 입력하세요.'))
y = int(input( '두번째정수를 입력하세요.'))
z = int(input( "세번째 정수를 입력하세요"))

if x % 2 == 0:
    print(x, '짝수')
else:
    print(x, '홀수')

if y % 2 == 0:
    print(y, '짝수')
else:
    print(y, '홀수')

if z % 2 == 0:
    print(z, '짝수')
else:
    print(z, '홀수')

x = int(input( '첫번째정수를 입력하세요.'))

if x >= 90:
    print('A학점')
elif x>80 and x < 90:
    print('B학점')
elif x>70 and x < 80:
    print('C학점')
elif x>60 and x < 70:
    print('D학점')
else:
    print('F학점')

x = input( '입력하세요.')

if x == 's':
    print('sieun')
elif x == 'm':
    print('makit')
elif x == 'j':
    print('james')
elif x == 'w':
    print('woojin')
else:
    print('howard')

num = int(input( '입력하세요.'))
if num == 12 or num == 1 or num == 2:
     print("겨울")
elif num == 3 or num == 4 or num == 5:
     print("봄")
elif num == 6 or num == 7 or num == 8:
     print("여름")
elif num == 9 or num == 10 or num ==11:
     print("가을")
else:
    print('해당 월은 없습니다.')

num = int(input( '입력하세요.'))
if num == 1 or num == 3 or num == 5 or num == 7 or num == 8 or num ==10 or num ==12 :
     print("31일까지 있습니다")
elif num == 4 or num == 6 or num == 9 or num == 11:
     print("30일까지 있습니다.")
elif num == 2:
     print("28일까지 있습니다")
else:
    print('해당 월은 없습니다.')

x = input( '이름을 입력하세요')
y = int(input( '키를 입력하세요.'))

if x[0] =='m':
    print('탈 수 있어요.')
else:
    if y >= 170:
        print('탈 수 있어요.')
    else:
        print('탈 수 없어요.')

x = int(input( '키를 입력하세요.'))

if x > 175:
    if x <180 :
        print('탈 수 있어요.')
    else:
        print('탈 수 없어요.')

elif x > 150:
    if x <170 :
        print('탈 수 있어요.')
    else:
        print('탈 수 없어요.')

else:
    print('탈 수 없어요.')

x = input( '성별을 입력하세요')
y = int(input( '키를 입력하세요.'))

if x == '여자':
    if y > 150 and y <170 :
        print('탈 수 있어요.')
    else:
        print('탈 수 없어요.')
elif y % 5 == 0:
     print('탈 수 있어요.')
else:
    print('탈 수 없어요.')

x = int(input( '첫번째정수를 입력하세요.'))
y = int(input( '두번째정수를 입력하세요.'))
z = int(input( "세번째 정수를 입력하세요"))
car = int(input( "차 높이를 입력하세요"))

if x > car and y > car and z > car:
    print('터널 통과 가능')
else:
    print('터널통과 불가능')

year = int(input( '첫번째정수를 입력하세요.'))

if year % 4 == 0:
    if year % 100 != 0:
        if year % 400 == 0:
            print("윤년입니다.")
        else :
            print("윤년이 아닙니다.")
    else :
        print("윤년이 아닙니다.")
else :
    print("윤년이 아닙니다.")

x = int(input( '첫번째정수를 입력하세요.'))

if num[7] == 3 or num[7] == 4:
   number = 20 + num[7:8]
else:
    number = 19 + num{7:8]

def parse_ssn(ssn):
    parts = ssn.split('-')
    year_prefix = '19' if parts[1][0] in ('1', '2') else '20'
    year = year_prefix + parts[0][:2]
    gender_code = parts[1][0]

    if gender_code in ('1', '3'):
        gender = '남자'
    elif gender_code in ('2', '4'):
        gender = '여자'
    else:
        gender = '알 수 없음'

    print(f'{gender} {year} 년 출생')

ssn_input = input('주민등록번호 입력:')
parse_ssn(ssn_input)

yut_input = input().split()
count_one = yut_input.count('1')


if num == 0:
    print('모')
elif num == 1:
    print('도')
elif num == 2:
    print('개')
elif num == 3:
    print('걸')
elif num == 4:
    print('윷')
else:
    pass

man_input = input("몸무게(kg)와 키(m)를 띄어쓰기로 구분하여 입력하세요: ").split()
weight = float(man_input[0])
height = float(man_input[1])

bmi = weight / height ** 2

if bmi < 18.5:
    print("저체중")
elif 18.5 <= bmi <= 22.9:
    print("정상")
elif 23 <= bmi <= 24.9:
    print("과체중")
elif 25 <= bmi <= 29.9:
    print("경도비만")
else:
    print("고도비만")

a = list(map(int, input().split()))
a.sort()

if a[2] > a[0] + a[1]:
    print("삼각형이 될 수 있습니다.")
else:
    print("삼각형이 될 수 없습니다.")



for i in range(10,-1,-1):
    print(i)
print('발사')

for i in range(11):
    print(i)
print('발사')

x = int(input( '카운트다운 몇 초 전인가요?'))

for i in range(x,-1,-1):
    print(i)
print('발사')

n = int(input())

sum = 0

for i in range(n):
    sum += i
    print(sum)

number = 1
sum = 0

while number <= 10:
    sum += number
    number += 1

print(f"1부터 10 까지 모두 더한 합은 {sum}")

while True:
    n = int(input("숫자를 입력하세요: "))
    if n < 0 :
        print("종료합니다")
        break
    print(f"입력된 숫자는 {n}")

print('1', end = ' ')
print('2', end = ' ')
print('3', end = ' ')
print('4', end = ' ')
print('5', end = ' ')
print('6', end = ' ')
print('7', end = ' ')
print('8', end = ' ')
print('9', end = ' ')
print('10', end = ' ')

n = int(input('첫번째 숫자는?'))
m = int(input('두번째 숫자는?'))

if n < m :
    sum = 0
    for i in range(n,m+1):
        sum += i
    print(sum)
else:
    print('첫번째 숫자가 두번째 숫자보다 작아야합니다,')

n = int(input('첫번째 숫자는?'))
m = int(input('두번째 숫자는?'))

if n < m :
    sum = 0
    if n%2==0:
        for i in range(n,m+1,2):
            sum += i
        print(sum)
    else:
        for i in range(n+1,m+1,2):
            sum += i
        print(sum)
else:
    print('첫번째 숫자가 두번째 숫자보다 작아야합니다,')

first_dice = range(1, 5)
second_dice = range(4, 8)

combinations = []

for first_roll in first_dice:
    for second_roll in second_dice:
        sum_of_rolls = first_roll + second_roll
        combinations.append((first_roll, second_roll, sum_of_rolls))

print("두 주사위를 던져서 나올 수 있는 숫자와 두 주사위의 합:")
for combination in combinations:
    print(f"첫 번째 주사위: {combination[0]}, 두 번째 주사위: {combination[1]}, 합: {combination[2]}")
print(f"가능한 눈의 경우의 수: {len(combinations)}")

print(256*256*256)

color_com = 0

for i in range(256):
    for j in range(256):
        for k in range(256):
            color_com += 1

color_com

for _ in range(4):
    for _ in range(4):
        print("*", end="")
    print()

for i in range(1,6):
    print(i*'*')

for i in range(5):
    for j in range(5):
        if i == j:
            print("*", end="")
        else:
            print(" ", end="")
    print()

rows = 5

for i in range(rows, 0, -1):
    print("*" * i)

rows = 3

for i in range(1, rows + 1):
    print(" " * (rows - i) + "*" * (2*i - 1) + " " * (rows - i))

rows = 5

for i in range(1, rows + 1):
    print(" " * (i - 1) + "*" + " " * (2 * (rows - i) - 1) + "*" + " " * (i - 1))

for i in range(rows, 0, -1):
    print(" " * (i - 1) + "*" + " " * (2 * (rows - i) - 1) + "*" + " " * (i - 1))

input_string = "apple makit woojin james sieun"
output_string = ""

for char in input_string:
    if char != 'a':
        output_string += char

print(output_string)

friends = ["우진", "시은", "메이킷", "지연", "지훈"]

for i in range(len(friends)):
    print(f"{i+1}번 친구 {friends[i]}을(를) 소개합니다.")

for i in range(30):
    if i%5 == 0:
        continue
    else:
        print(i, end = ' ')

start = 3
increment = 5
target_index = 100
result = 0

current_number = start
for i in range(1, target_index + 1):
    if i == target_index:
        result = current_number
        break
    current_number += increment

print(result)



x = int(input( '첫번째정수를 입력하세요.'))
y = int(input( '두번째정수를 입력하세요.'))
z = int(input( '세번째정수를 입력하세요.'))


min(x,y,z)

x = int(input( '첫번째정수를 입력하세요.'))
y = int(input( '두번째정수를 입력하세요.'))
z = int(input( '세번째정수를 입력하세요.'))


if x < y:
    if x < z:
        print(x)
    else:
        print(z)
else:
    if y < z:
        print(y)
    else:
        print(z)

x = int(input( '첫번째정수를 입력하세요.'))
a = list(map(int, input().split()))

if x < a[0]:
    if x < a[1]:
        if x < a[2]:
            print("통과 가능")
        else:
            print("통과 불가능")
    else:
        print("통과 불가능")
else:
    print("통과 불가능")

count = 0
for num in range(1, 1001):
    count += str(num).count('7')

print("1부터 1000까지의 숫자 중에서 행운의 숫자 7은 총", count, "번 사용되었습니다.")

n = int(input("자연수를 입력하세요: "))

print("양의 약수(오름차순):", end=" ")
for i in range(1, n + 1):
    if n % i == 0:
        print(i, end=" ")

print("\n양의 약수(내림차순):", end=" ")
for i in range(n, 0, -1):
    if n % i == 0:
        print(i, end=" ")

n = int(input("정수의 개수를 입력하세요: "))
numbers = list(map(int, input("정수를 공백으로 구분하여 입력하세요: ").split()))

max_value = max(numbers)
min_value = min(numbers)

print("리스트 최댓값:", max_value)
print("리스트 최솟값:", min_value)

number = input("자연수를 입력하세요: ")
total = 0

for digit in number:
    total += int(digit)

print("각 자릿수의 합:", total)

number = int(input("자연수를 입력하세요: "))
total = 0

while number > 0:
    total += number % 10
    number //= 10

print("각 자릿수의 합:", total)

text = input("문자열을 입력하세요: ")
encrypted_text = ""

for char in text:
    if 'A' <= char <= 'Z':
        encrypted_text += chr((ord(char) - ord('A') + 3) % 26 + ord('A'))
    elif 'a' <= char <= 'z':
        encrypted_text += chr((ord(char) - ord('a') + 3) % 26 + ord('a'))
    else:
        encrypted_text += char

print("암호문:", encrypted_text)

sentence = input("영어 문장을 입력하세요: ")
converted_sentence = ""

for char in sentence:
    if 'A' <= char <= 'Z':
        converted_sentence += char.lower()
    elif 'a' <= char <= 'z':
        converted_sentence += char.upper()
    else:
        converted_sentence += char

print("변환된 문장:", converted_sentence)



store = {'메이킷': 17, '우진': 9, '시은': 11, '제임스': 10}

print(store)

s = {
    '사는 곳': '서울',
    '성별': '여자',
    '이름': '시은',
    '나이': 11,
    '혈액형': 'b'
}

print("내가 살고 있는 곳은", s['사는 곳'], "입니다.")
print("나의 성별은", s['성별'], "이고 이름은", s['이름'], "입니다.")
print("나이는", s['나이'], "살 이고 혈액형은", s['혈액형'], "형입니다.")

s = {
    '사는 곳': '인천 송도',
    '성별': '남자',
    '이름': '우진'
}

age = input("나이는 몇 살인가요? ")
blood_type = input("혈액형은 무엇인가요? ")

s['나이'] = age
s['혈액형'] = blood_type

print("나의 성별은", s['성별'], "이고 이름은", s['이름'], "입니다.")
print("나이는", s['나이'], "살 이고 혈액형은", s['혈액형'], "형입니다.")

s = {'사는 곳': '인천 송도', '성별': '남자', '이름': '우진'}

s.update({'나이': '9', '혈액형': 'b'})

print(s)

s = {'사는 곳': '인천 송도', '성별': '남자', '이름': '우진'}

a = list(s.keys())

b = list(s.values())

c = list(s.items())

print(a)
print(b)
print(c)

store = {'새우칩': 1500, '옥수수칩': 1800, '콜라': 700, '양파칩': 1300}

print("물품을 입력하면 가격을 알려드립니다")

while True:
    item_name = input("물품 이름은? ")
    if item_name in store:
        print(item_name + "의 가격은", store[item_name], "입니다")
    else:
        print(item_name + "는 등록되어 있지 않습니다")
        break

store = {'새우칩': 1500, '옥수수칩': 1800, '콜라': 700, '양파칩': 1300}

total_price = 0
item_count = 0

print("물품을 입력하면 가격을 알려드립니다")

for item, price in store.items():
    item_count += 1
    total_price += price
    print(f"{item_count} 번 물품은 {item} 이고 가격은 {price} 입니다")

print(f"우리 가게는 모두 {item_count} 개 종류의 물건이 있습니다")
print(f"우리 가게 물건들의 가격 총합은 {total_price} 입니다")

store = {'새우칩': {'가격': 1500, '수량': 3},
         '옥수수칩': {'가격': 1800, '수량': 2},
         '콜라': {'가격': 700, '수량': 4},
         '양파칩': {'가격': 1300, '수량': 1}}

total_price = 0
total_quantity = 0
item_count = 0

print("물품을 입력하면 가격을 알려드립니다")

for item, info in store.items():
    item_count += 1
    total_price += info['가격'] * info['수량']
    total_quantity += info['수량']
    print(f"{item_count} 번 물품은 {item} 이고 가격은 {info['가격']} 이고 수량은 {info['수량']} 개입니다")

print(f"우리 가게는 {item_count} 개의 물건 종류가 있습니다")
print(f"우리 가게 모든 물건 수량은 {total_quantity} 개 있습니다")
print(f"우리 가게 모든 물건들의 가격 총합은 {total_price} 입니다")
print(f"우리 가게 모든 물건들의 평균 가격은 {total_price / total_quantity} 원입니다")

scores = {'우진': {'코딩': 95},
          '시은': {'과학': 100},
          '메이킷': {'영어': 85}}

print(f"우진이의 코딩 점수는? {scores['우진'].get('코딩')}")
print(f"시은이의 과학 점수는? {scores['시은'].get('과학')}")
print(f"메이킷의 영어 점수는? {scores['메이킷'].get('영어')}")



a = [1, 2, 1, 1, 2, 3, 4, 1, 5, 2, 1, 5]

unique_values = []
for num in a:
    if num not in unique_values:
        unique_values.append(num)

print(unique_values)

a = {'메이킷', '우진', '시은', '소피아'}

a.add('하워드')
print(a)

a.remove('소피아')
print(a)

travel_info = {
    '하와이': {'우진', '윤진', '시은'},
    '이탈리아': {'윤진', '시은', '형우'},
    '두바이': {'시은', '우진', '메이킷'}
}

all_travelers = set()
for travelers in travel_info.values():
    if not all_travelers:
        all_travelers = travelers
    else:
        all_travelers = all_travelers.intersection(travelers)
print(all_travelers)

hawaii_or_italia_travelers = set()
for travelers in travel_info.values():
    hawaii_or_italia_travelers = hawaii_or_italia_travelers.union(travelers)
print("하와이 또는 이탈리아 여행을 다녀온 사람은?")
print(hawaii_or_italia_travelers)

dubai_only_travelers = travel_info['두바이']
for travelers in travel_info.values():
    if travelers == dubai_only_travelers:
        continue
    dubai_only_travelers = dubai_only_travelers.difference(travelers)
print("두바이 여행은 다녀왔고 하와이와 이탈리아 여행 경험이 없는 사람은?")
print(dubai_only_travelers)

hawaii = {'우진', '윤진', '시은'}
italia = {'윤진', '시은', '형우'}
dubai = {'시은', '우진', '메이킷'}

all_travelers = hawaii.intersection(italia, dubai)
print("하와이, 이탈리아, 두바이 모두 여행을 다녀온 사람은?")
print(all_travelers)

hawaii_or_italia_travelers = hawaii.union(italia)
print("하와이 또는 이탈리아 여행을 다녀온 사람은?")
print(hawaii_or_italia_travelers)

dubai_only_travelers = dubai.difference(hawaii, italia)
print("두바이 여행은 다녀왔고 하와이와 이탈리아 여행 경험이 없는 사람은?")
print(dubai_only_travelers)

def makit(n):
    total_sum = 0
    for i in range(1, n+1):
        total_sum += i
    print(f"1부터 {n}까지 합은 {total_sum} 입니다.")

makit(10)
makit(100)

# 문제 093: 함수 값 돌려주기(return)
# 두 개의 자연수를 n1, n2를 입력받습니다. n2는 n1보다 큰 숫자입니다. 아래와 같은 결과가 나오도록 n1, n2를 사용자 정의 함수 makit에 전달한 뒤 n1부터 n2까지의 합을 구해 다시 함수를 호출한 곳으로 돌려 보내는 코드를 작성해 보세요.

# 실행 결과
# 첫 번째 숫자를 입력하세요1
# 두 번째 숫자를 입력하세요10
# 1 + ... + 10 = 55

# 코드
# def makit(n1, n2):

# n1 = int(input('첫 번째 숫자를 입력하세요'))
# n2 = int(input('두 번째 숫자를 입력하세요'))

# print(n1, '+ ... +', n2, '=', makit(n1,n2))

def makit(n1, n2):
    x = makit_abs(n1)
    y = makit_abs(n2)

    if x > y:
        print(f"절댓값이 큰 수는: {n1}")
    else:
        print(f"절댓값이 큰 수는: {n2}")

def makit_abs(n):
    if n < 0:
        return -n
    else:
        return n

makit(11, -13)

left = ['초록', '초록', '빨강', '노랑', '노랑', '파랑', '남색', '보라']
right = ['파랑', '초록', '초록', '보라', '보라', '노랑', '빨강', '빨강', '파랑']

def makit(left, right):
    color = ['빨강', '주황', '노랑', '초록', '파랑', '남색', '보라']

    left_num = [0, 0, 0, 0, 0, 0, 0]
    right_num = [0, 0, 0, 0, 0, 0, 0]

    for color_index in range(len(color)):
        left_num[color_index] = left.count(color[color_index])
        right_num[color_index] = right.count(color[color_index])

    total = 0
    for i in range(len(color)):
        print(f"{color[i]} 색으로 만들 수 있는 장갑은 {min(left_num[i], right_num[i])} 개")
        total += min(left_num[i], right_num[i])

    print(f"최대로 만들 수 있는 장갑 쌍은 {total} 개")
    return total

makit(left, right)

# 문제 096: 괄호 검사기 만들기
# 괄호 검사기를 만들려고 합니다. 괄호 검사기는 다음과 같이 여는 괄호와 닫는 괄호를 순서와 개수를 만족할 때 성공이고, 그렇지 못한 경우에는 실패입니다.

# (())는 성공
# (((()))는 닫는 괄호가 한 개 마지막에 모자라므로 실패
# )()(는 여는 괄호가 없이 먼저 닫는 괄호로 시작했으므로 실패
# (()))는 닫는 괄호가 한 개 많으므로 실패
# ()()()는 성공
# 괄호의 자료가 입력 문자열로 주어지면, 성공과 실패를 판단하는 함수를 포함하는 코드를 작성하려고 합니다. 다음과 같이 결과를 출력하도록 코드를 완성하세요.

# 실행 결과 예시1
# 괄호의 자료를 입력하세요:()()()
# 성공
# 실행 결과 예시2
# 괄호의 자료를 입력하세요:(()))
# 실패
# 실행 결과 예시3
# 괄호의 자료를 입력하세요:)()(
# 실패

# 코드
# n = input('괄호의 자료를 입력하세요:')
# def makit(n):

# if makit(n): # 괄호 검사 함수 호출
#     print('성공')
# else:
#     print('실패')

a = [12, 23, 1, 15, 75, 79, 19]

a.sort()
print(a)

b = [12, 23, 1, 15, 75, 79, 19]

b.sort(reverse=True)
print(b)

a = [12, 23, 1, 15, 75, 79, 19]
b = sorted(a)
print(b)
print(a)

c = [12, 23, 1, 15, 75, 79, 19]
c.sort()
print(c)
d = sorted(c, reverse=True)
print(d)

name = ['메이킷', '소피아', '우진', '시은', '하워드']
score = [92, 75, 95, 96, 89]

n = int(input('몇 등 학생 자료를 알고 싶나요? '))

sorted_scores = sorted(score, reverse=True)

student_score = sorted_scores[n - 1]
idx = score.index(student_score)

print(n, '등 학생은', student_score, '점이고 이름은', name[idx], '입니다')

a = [13, 7, 2, 199, 24, 5]

def makit_selection(a):
    n = len(a)

    for i in range(n - 1):
        min_idx = i
        for j in range(i + 1, n):
            if a[j] < a[min_idx]:
                min_idx = j
        a[i], a[min_idx] = a[min_idx], a[i]

        print(a)

    return a

print('최종 정렬 결과', makit_selection(a))

