import json


def load_json_data(filepath):
    try:  # 读取json文件
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except (FileNotFoundError, json.JSONDecodeError) as e:  # 捕获异常
        print(f"Error: {e}")
        return None


def get_bmi(bmi):
    if bmi < 18.5:
        return 'UnderWeight'
    elif bmi < 24:
        return 'NorWeight'
    elif bmi < 28:
        return 'OverWeight'
    else:
        return 'Obesity'


def get_adv2(pregnancy, bmi, er, week, grade, unit):
    if week <= 12:
        range = data.get(pregnancy).get(str(week))
    else:
        range = data.get(pregnancy).get(str(week)).get(str(bmi))
    low = range.get('low')
    high = range.get('high')
    food_range = food.get(pregnancy).get(str(grade)).get(str(unit))
    food_low = food_range.get('low')
    food_high = food_range.get('high')
    wtida = int(((er - low) / (high - low)) * 100)
    # print(er, range, food_range)
    if wtida == 0:
        wtida = 0.003
    if wtida <= 90 and wtida >= 0:
        return 2, 0, food_range
    elif wtida > 90 and wtida <= 120:
        return 1, food_low, food_range
    elif wtida > 120 and wtida <= 150:
        return 1, food_high, food_range
    elif wtida >= -20:
        return 2, 0, food_range
    elif wtida <= -20 and wtida >= -50:
        return 0, food_low, food_range
    elif wtida >= -50:
        return 4, -1, food_range
    elif wtida >= 150:
        return 4, -1, food_range
    else:
        return 4, -1, food_range


def get_adv(pregnancy, bmi, er, week, grade, unit):
    if week <= 12:
        range = data.get(pregnancy).get(str(week))
    else:
        range = data.get(pregnancy).get(str(week)).get(str(bmi))
    low = range.get('low')
    high = range.get('high')
    food_range = food.get(pregnancy).get(str(grade)).get(str(unit))
    food_low = food_range.get('low')
    food_high = food_range.get('high')
    er_50 = (high - low) * -0.5 + low
    er_20 = (high - low) * -0.2 + low
    er90 = (high - low) * 0.9 + low
    er120 = (high - low) * 1.2 + low
    er150 = (high - low) * 1.5 + low
    # print(er, range, food_range)
    if er < er_50:
        return 1, food_range
    elif er < er_20:
        return food_low, food_range
    elif er < er90:
        return 0, food_range
    elif er < er120:
        return -food_low, food_range
    elif er < er150:
        return -food_high, food_range
    else:
        return -1, food_range


def get_user_weight(week, weight, preweight, bmi, grade="mode", unit="kcal"):
    bmi = get_bmi(bmi)
    er = round(weight - preweight, 2)

    if week <= 12:
        return get_adv("prePregnancy", bmi, er, week, grade, unit)
    elif week < 28:
        return get_adv("midPregnancy", bmi, er, week, grade, unit)
    else:
        return get_adv("latePregnancy", bmi, er, week, grade, unit)


def get_user_weight1(week, weight, preweight, bmi):
    grade = "mode"
    unit = "kcal"
    bmi = get_bmi(bmi)
    er = round(weight - preweight, 2)

    if week <= 12:
        return get_adv2("prePregnancy", bmi, er, week, grade, unit)
    elif week < 28:
        return get_adv2("midPregnancy", bmi, er, week, grade, unit)
    else:
        return get_adv2("latePregnancy", bmi, er, week, grade, unit)


# 建议使用绝对路径来指定文件位置
data = load_json_data("./nutrition_data/weight_std.json")
food = load_json_data("./nutrition_data/weight_food.json")

if __name__ == '__main__':
    # ude,adv, ce = get_user_weight(28, 86.5, 80, 27, "mode", "kcal")
    adv, ce = get_user_weight(40, 999, 80, 999)

    print("adv", adv)
    print(ce)
