import json
from typing import List, Tuple, Optional
import math
import pandas as pd
from pandas import DataFrame

from diet_recom.src.weight_gain_food import get_user_weight


# 基础数据类定义
class PregnancyInfo:
    """孕妇基础信息类"""

    def __init__(self, bmi: float, week: int, trimester: int, pre_pregnancy_weight: float, pregnancy_weight: float,
                 activity_level: float):
        self.bmi = bmi  # 孕前BMI
        self.week = min(week, 40) if week else 1  # 当前孕周
        self.trimester = trimester  # 孕期：1=孕早期，2=孕中期，3=孕晚期
        self.pre_pregnancy_weight = pre_pregnancy_weight  # 孕前体重(kg)
        self.pregnancy_weight = pregnancy_weight  # 孕前体重(kg)
        self.activity_level = activity_level  # 活动系数：1.2(久坐)、1.375(轻度)、1.55(中度)、1.725(重度)


class NutritionRequirements:
    """营养需求类"""

    def __init__(self, total_energy: float, protein_amount: float, min_carb: float, max_carb: float,
                 min_fat: float, max_fat: float, carb_percentage: Tuple[float, float],
                 fat_percentage: Tuple[float, float]):
        self.total_energy = total_energy  # 每日总能量(kcal)
        self.protein_amount = protein_amount  # 每日蛋白质总量(g)
        self.min_carb = min_carb  # 每日碳水化合物最小值(g)
        self.max_carb = max_carb  # 每日碳水化合物最大值(g)
        self.min_fat = min_fat  # 每日脂肪最小值(g)
        self.max_fat = max_fat  # 每日脂肪最大值(g)
        self.carb_percentage = carb_percentage  # 碳水能量占比范围
        self.fat_percentage = fat_percentage  # 脂肪能量占比范围


class Dish:
    """菜品类（适配Excel数据）"""

    def __init__(self, dish_id: int, name: str, category: str, core_category: str, composition: str, serving_size: str,
                 energy: float, protein: float, fat: float, carbohydrate: float, cooking_method: str, recom_meal: str,
                 icon: str):
        self.dish_id = dish_id  # 菜品ID（对应Excel的dish_id）
        self.name = name  # 菜品名称（对应Excel的dish_name）
        self.category = category  # 菜品分类（对应Excel的category）
        self.core_category = core_category  # 菜品核心分类（对应Excel的core_category）
        self.composition = composition  # 食材组成（对应Excel的composition）
        self.serving_size = serving_size  # 单位份量（对应Excel的serving_size）
        self.energy = energy  # 能量(kcal/100g)（对应Excel的energy）
        self.protein = protein  # 蛋白质(g/100g)（对应Excel的protein）
        self.fat = fat  # 脂肪(g/100g)（对应Excel的fat）
        self.carbohydrate = carbohydrate  # 碳水化合物(g/100g)（对应Excel的carbohydrates）
        self.cooking_method = cooking_method  # 烹饪方式（对应Excel的cooking_method）
        self.recom_meal = recom_meal  # 推荐餐次（对应Excel的recom_meal）
        self.icon = icon  # 餐品icon


class Serving:
    """菜品份量类"""

    def __init__(self, dish: Dish, grams: float):
        self.dish = dish
        self.grams = grams  # 食用克重
        # 计算该份量的营养值
        self.energy = (dish.energy / 100) * grams
        self.protein = (dish.protein / 100) * grams
        self.fat = (dish.fat / 100) * grams
        self.carbohydrate = (dish.carbohydrate / 100) * grams


class MealNutrition:
    """单餐营养目标类"""

    def __init__(self, meal_name: int, energy: float, protein: float, fat: float, carbohydrate: float):
        self.meal_name = meal_name  # 餐次名称(早餐/午餐/晚餐)
        self.energy = energy  # 单餐能量目标(kcal)
        self.protein = protein  # 单餐蛋白质目标(g)
        self.fat = fat  # 单餐脂肪目标(g)
        self.carbohydrate = carbohydrate  # 单餐碳水目标(g)


class RecommendedMeal:
    """推荐单餐类"""

    def __init__(self, meal_name: int, servings: List[Serving], total_energy: float, total_protein: float,
                 total_fat: float, total_carbohydrate: float, target_energy: float, target_protein: float,
                 target_fat: float, target_carbohydrate: float):
        self.meal_name = meal_name
        self.servings = servings  # 菜品份量列表
        self.total_energy = total_energy  # 实际总能量
        self.total_protein = total_protein  # 实际总蛋白质
        self.total_fat = total_fat  # 实际总脂肪
        self.total_carbohydrate = total_carbohydrate  # 实际总碳水
        # 目标营养值（用于对比显示）
        self.target_energy = target_energy
        self.target_protein = target_protein
        self.target_fat = target_fat
        self.target_carbohydrate = target_carbohydrate


class DailyDietPlan:
    """单日饮食计划类（含3套方案）"""

    def __init__(self, day: int, daily_nutrition: NutritionRequirements,
                 plans: List[List[RecommendedMeal]], used_dish_ids: List[int]):
        self.day = day  # 第几天
        self.daily_nutrition = daily_nutrition  # 当日营养需求
        self.plans = plans  # 3套方案，每套含早午晚3餐（plans[0]=A方案，plans[1]=B方案，plans[2]=C方案）
        self.used_dish_ids = used_dish_ids  # 当日使用的菜品ID（避免重复）


class WeeklyDietPlan:
    """一周饮食计划类"""

    def __init__(self, info: PregnancyInfo, daily_plans: List[DailyDietPlan]):
        self.pregnancy_info = info  # 孕妇信息
        self.daily_plans = daily_plans  # 每日计划列表
        self.total_days = len(daily_plans)  # 总天数


# 孕妇饮食推荐核心类
class PregnancyDietRecommender:
    def __init__(self, dishes: List[Dish]):
        self.dishes = dishes  # 全部菜品列表
        self.all_dish_ids = [d.dish_id for d in dishes]  # 全部菜品ID
        # 按分类分组（适配Excel分类：主食类/畜肉类/禽肉类/水产品类/蛋奶豆类/素菜类/汤羹类）
        self.dishes_by_category = self._group_dishes_by_category()
        # 基础配置
        self.min_dishes_per_meal = 3  # 每餐最少菜品数
        self.max_dishes_per_meal = 4  # 每餐最多菜品数
        self.serving_size_range = (50.0, 250.0)  # 单菜品份量范围(克)
        # 每日3套方案的菜品偏好（A=经典家常，B=清淡快手，C=高纤营养）
        self.plan_preferences = {
            "A": {"prefer_categories": ["主食类", "禽肉类", "畜肉类", "素菜类"], "avoid_fat": False},
            "B": {"prefer_categories": ["主食类", "水产品类", "汤羹类", "素菜类"], "avoid_fat": True},
            "C": {"prefer_categories": ["主食类", "蛋奶豆类", "素菜类", "水产品类"], "high_fiber": True}
        }

    def _group_dishes_by_category(self) -> dict:
        """按菜品分类分组"""
        category_dict = {}
        for dish in self.dishes:
            if dish.core_category in ["其他"]:
                continue
            if dish.core_category not in category_dict:
                category_dict[dish.core_category] = []
            category_dict[dish.core_category].append(dish)
        return category_dict

    def determine_nutrition_requirements(self, info: PregnancyInfo, is_male: bool = False) -> NutritionRequirements:
        """
        参考《中国居民膳食营养素参考摄入》计算基础代谢率(BMR)
        注意：这里假设使用者为女性，因为是孕妇饮食推荐
        """
        user_sex = 0 if is_male else 1
        base_energy = 14.52 * info.pre_pregnancy_weight - 155.88 * user_sex + 565.79
        adv, ce = get_user_weight(info.week, info.pregnancy_weight, info.pre_pregnancy_weight, info.bmi)
        # if adv == -1:
        #     raise ReferenceError("您的体重已超出可控的增长范围，请去查看医生")
        adv_energy = 0
        try:
            if adv == -1:
                adv_energy = -ce.get('high')
            elif adv == 1:
                adv_energy = ce.get('high')
            else:
                adv_energy = adv
        except Exception as e:
            adv_energy = 0
        # 2. 总能量 = BEE × 活动系数 + 孕期增量
        trimester_energy = {1: 60, 2: 276, 3: 418}  # 孕早/中/晚能量增量(kcal)
        total_energy = base_energy * info.activity_level + trimester_energy[info.trimester]
        total_energy = total_energy + adv_energy
        total_energy = round(total_energy, 1)

        # 3. 蛋白质需求（孕中+15g，孕晚+30g；孕前基础：0.8g/kg体重）
        pre_protein = info.pre_pregnancy_weight * 0.8
        if info.trimester == 1:
            protein_amount = pre_protein
        elif info.trimester == 2:
            protein_amount = pre_protein + 15
        else:
            protein_amount = pre_protein + 30
        protein_amount = round(protein_amount, 1)

        # 计算当前蛋白质对应的能量
        current_protein_energy = protein_amount * 4
        # 计算总能量的15%~20%范围
        min_protein_energy = total_energy * 0.15
        max_protein_energy = total_energy * 0.20
        # 统一保留1位小数，消除浮点精度偏差
        current_protein_energy = round(current_protein_energy, 1)
        min_protein_energy = round(min_protein_energy, 1)
        max_protein_energy = round(max_protein_energy, 1)
        # 若低于15%，按15%补全；若高于20%，按20%限制
        if current_protein_energy < min_protein_energy:
            # 按总能量15%反推最小蛋白质需求量
            required_protein = min_protein_energy / 4
            # 确保不低于孕期最低要求（孕中≥pre_protein+15，孕晚≥pre_protein+30）
            if info.trimester == 2:
                required_protein = max(required_protein, pre_protein + 15)
            elif info.trimester == 3:
                required_protein = max(required_protein, pre_protein + 30)
            protein_amount = round(required_protein, 1)
        elif current_protein_energy > max_protein_energy:
            # 按总能量20%反推最大蛋白质需求量
            required_protein = max_protein_energy / 4
            protein_amount = round(required_protein, 1)
        # 验证蛋白质能量占比（15%-20%）
        # current_protein_energy = protein_amount * 4
        # assert (current_protein_energy >= min_protein_energy - 0.1) and (
        #         current_protein_energy <= max_protein_energy + 0.1), \
        #     f"蛋白质能量占比不符合要求（总能量：{total_energy}kcal）\n" \
        #     f"要求范围：{min_protein_energy}-{max_protein_energy}kcal（总能量15%-20%）\n" \
        #     f"当前值：{current_protein_energy}kcal（蛋白质摄入量：{protein_amount}g）"

        # 4. 碳水化合物需求（50%-65%能量，≥130g）
        min_carb_energy = total_energy * 0.5
        max_carb_energy = total_energy * 0.65
        min_carb = max(130.0, round(min_carb_energy / 4, 1))  # 1g碳水=4kcal
        max_carb = round(max_carb_energy / 4, 1)

        # 5. 脂肪需求（20%-30%能量）
        min_fat_energy = total_energy * 0.2
        max_fat_energy = total_energy * 0.3
        min_fat = round(min_fat_energy / 9, 1)  # 1g脂肪=9kcal
        max_fat = round(max_fat_energy / 9, 1)

        return NutritionRequirements(
            total_energy=round(total_energy),
            protein_amount=protein_amount,
            min_carb=min_carb,
            max_carb=max_carb,
            min_fat=min_fat,
            max_fat=max_fat,
            carb_percentage=(0.5, 0.65),
            fat_percentage=(0.2, 0.3)
        )

    def split_nutrition_to_meals(self, requirements: NutritionRequirements) -> List[MealNutrition]:
        """
        按早40%/中40%/晚20%拆分营养到三餐
        """
        meals = []
        # 早餐：40%能量
        breakfast_energy = requirements.total_energy * 0.4
        breakfast_protein = requirements.protein_amount * 0.4
        breakfast_carb = ((requirements.min_carb + requirements.max_carb) / 2) * 0.4
        breakfast_fat = ((requirements.min_fat + requirements.max_fat) / 2) * 0.4
        meals.append(MealNutrition(
            meal_name=1,
            energy=round(breakfast_energy, 1),
            protein=round(breakfast_protein, 1),
            fat=round(breakfast_fat, 1),
            carbohydrate=round(breakfast_carb, 1)
        ))

        # 午餐：35%能量
        lunch_energy = requirements.total_energy * 0.4
        lunch_protein = requirements.protein_amount * 0.4
        lunch_carb = ((requirements.min_carb + requirements.max_carb) / 2) * 0.4
        lunch_fat = ((requirements.min_fat + requirements.max_fat) / 2) * 0.4
        meals.append(MealNutrition(
            meal_name=2,
            energy=round(lunch_energy, 1),
            protein=round(lunch_protein, 1),
            fat=round(lunch_fat, 1),
            carbohydrate=round(lunch_carb, 1)
        ))

        # 晚餐：25%能量（脂肪适当降低10%，避免睡前负担）
        dinner_energy = requirements.total_energy * 0.2
        dinner_protein = requirements.protein_amount * 0.2
        dinner_carb = ((requirements.min_carb + requirements.max_carb) / 2) * 0.2
        dinner_fat = ((requirements.min_fat + requirements.max_fat) / 2) * 0.2 * 0.9
        meals.append(MealNutrition(
            meal_name=3,
            energy=round(dinner_energy, 1),
            protein=round(dinner_protein, 1),
            fat=round(dinner_fat, 1),
            carbohydrate=round(dinner_carb, 1)
        ))

        return meals

    def _filter_available_dishes(self, all_dishes: List[Dish], used_ids: List[int], plan_type: str, meal_name: int) -> \
            List[Dish]:
        """
        过滤已使用菜品+按方案偏好筛选
        plan_type: A/B/C，对应不同菜品偏好
        param meal_name: 当前推荐的餐次名称（1=早餐，2=午餐，3=晚餐），对应MealNutrition的meal_name字段
        """
        # 1. 排除已使用菜品
        available = [d for d in all_dishes if d.dish_id not in used_ids and d.core_category != "其他"]
        if not available:
            print(f"警告：无可用菜品（排除其他分类后，餐次：{meal_name}，方案：{plan_type}）")
            return []

        # 2. 按推荐餐次筛选（核心新增逻辑）
        # 映射meal_name（1/2/3）到对应餐次字符串
        meal_map = {1: "早餐", 2: "午餐", 3: "晚餐"}
        target_meal = meal_map.get(meal_name, "通用")  # 当前目标餐次（如早餐）
        # 筛选规则：recom_meal为目标餐次 或 "通用"（兼容标注为“通用”的菜品）
        available = [
            d for d in available
            if d.recom_meal == target_meal or d.recom_meal == "通用"
        ]
        if not available:
            # 若目标餐次无可用菜品，降级使用“通用”菜品
            available = [d for d in all_dishes if d.dish_id not in used_ids and d.recom_meal == "通用"]
            if not available:
                return []

        # 2. 按方案偏好筛选
        preference = self.plan_preferences[plan_type]
        if preference.get("avoid_fat"):
            # 清淡方案：排除高脂肪菜品（脂肪>5g/100g）
            available = [d for d in available if d.fat <= 5.0]
        if preference.get("high_fiber"):
            # 高纤方案：优先高碳水/高纤维菜品（碳水>10g/100g 或 分类含"素菜类"）
            available = [d for d in available if d.carbohydrate > 10.0 or d.core_category == "素菜类"]
        # 优先推荐偏好分类的菜品
        prefer_dishes = [d for d in available if d.core_category in preference["prefer_categories"]]
        return prefer_dishes if prefer_dishes else available

    def _calculate_optimal_serving(self, dish: Dish, meal: MealNutrition, remaining_energy: float) -> float:
        """计算菜品最佳食用克重（按剩余能量分配）"""
        if dish.energy <= 0:
            return 100.0  # 无能量数据菜品默认100g
        # 基础克重 = 剩余能量 / 菜品能量 * 100
        base_grams = (remaining_energy / dish.energy) * 100
        # 限制在份量范围内，且按50g整数倍取整（便于实际操作）
        min_grams, max_grams = self.serving_size_range
        optimal_grams = max(min_grams, min(base_grams, max_grams))
        return round(optimal_grams / 50) * 50

    def _recommend_single_meal(self, meal: MealNutrition, used_ids: List[int], plan_type: str) -> Tuple[
        RecommendedMeal, List[int]]:
        """为单餐推荐菜品（按方案类型）"""
        remaining_energy = meal.energy
        remaining_protein = meal.protein
        remaining_fat = meal.fat
        remaining_carb = meal.carbohydrate
        selected_servings = []
        current_used_ids = []
        num_dishes = math.floor((self.min_dishes_per_meal + self.max_dishes_per_meal) / 2)  # 每餐3-4道，取中间值3道

        # 核心菜品分类组（确保营养均衡：主食+蛋白+蔬菜）
        core_category_groups = [
            ["主食类"],  # 必选：碳水来源
            ["禽肉类", "畜肉类", "水产品类", "蛋奶豆类"],  # 必选：蛋白来源
            ["素菜类", "汤羹类"]  # 必选：蔬菜/汤羹
        ]

        # 1. 优先选择核心分类菜品
        for group in core_category_groups:
            # 获取该组可用菜品（按方案偏好过滤）
            group_dishes = []
            for cat in group:
                if cat not in self.dishes_by_category:
                    continue
                cat_dishes = self._filter_available_dishes(
                    self.dishes_by_category[cat], used_ids + current_used_ids, plan_type, meal.meal_name
                )
                group_dishes.extend(cat_dishes)
            if not group_dishes:
                continue

            # 评分筛选：营养差值最小为最佳（能量+蛋白+脂肪+碳水）
            scored_dishes = []
            for dish in group_dishes:
                ideal_grams = self._calculate_optimal_serving(
                    dish, meal, remaining_energy / (num_dishes - len(selected_servings))
                )
                # 计算营养差值（越小越好）
                score = (
                        abs((dish.energy / 100 * ideal_grams) - remaining_energy / (
                                num_dishes - len(selected_servings))) +
                        abs((dish.protein / 100 * ideal_grams) - remaining_protein / (
                                num_dishes - len(selected_servings))) +
                        abs((dish.fat / 100 * ideal_grams) - remaining_fat / (num_dishes - len(selected_servings))) +
                        abs((dish.carbohydrate / 100 * ideal_grams) - remaining_carb / (
                                num_dishes - len(selected_servings)))
                )
                scored_dishes.append((dish, score))

            # 选择评分最低的菜品
            best_dish = min(scored_dishes, key=lambda x: x[1])[0]
            grams = self._calculate_optimal_serving(
                best_dish, meal, remaining_energy / (num_dishes - len(selected_servings))
            )
            serving = Serving(best_dish, grams)

            # 更新选择与剩余营养
            selected_servings.append(serving)
            current_used_ids.append(best_dish.dish_id)
            remaining_energy -= serving.energy
            remaining_protein -= serving.protein
            remaining_fat -= serving.fat
            remaining_carb -= serving.carbohydrate

            if len(selected_servings) >= num_dishes:
                break

        # 2. 补充菜品至最少数量（若核心分类不足）
        while len(selected_servings) < self.min_dishes_per_meal:
            all_available = self._filter_available_dishes(
                [d for cats in self.dishes_by_category.values() for d in cats],
                used_ids + current_used_ids,
                plan_type, meal.meal_name
            )
            if not all_available:
                break
            # 选择营养最接近剩余需求的菜品
            best_dish = min(
                all_available,
                key=lambda d: (
                        abs((d.energy / 100 * 100) - remaining_energy) +
                        abs((d.protein / 100 * 100) - remaining_protein) +
                        abs((d.fat / 100 * 100) - remaining_fat) +
                        abs((d.carbohydrate / 100 * 100) - remaining_carb)
                )
            )
            grams = self._calculate_optimal_serving(best_dish, meal, remaining_energy)
            serving = Serving(best_dish, grams)
            selected_servings.append(serving)
            current_used_ids.append(best_dish.dish_id)
            remaining_energy -= serving.energy
            remaining_protein -= serving.protein
            remaining_fat -= serving.fat
            remaining_carb -= serving.carbohydrate

        # 计算实际总营养
        total_energy = sum(s.energy for s in selected_servings)
        total_protein = sum(s.protein for s in selected_servings)
        total_fat = sum(s.fat for s in selected_servings)
        total_carb = sum(s.carbohydrate for s in selected_servings)

        return (
            RecommendedMeal(
                meal_name=meal.meal_name,
                servings=selected_servings,
                total_energy=round(total_energy),
                total_protein=round(total_protein, 1),
                total_fat=round(total_fat, 1),
                total_carbohydrate=round(total_carb, 1),
                target_energy=round(meal.energy),
                target_protein=meal.protein,
                target_fat=meal.fat,
                target_carbohydrate=meal.carbohydrate
            ),
            current_used_ids
        )

    def recommend_daily_plan(self, info: PregnancyInfo, used_ids: List[int]) -> DailyDietPlan:
        """生成单日3套饮食计划"""
        daily_nutrition = self.determine_nutrition_requirements(info)
        meal_nutrition_list = self.split_nutrition_to_meals(daily_nutrition)
        daily_plans = []  # 存储A/B/C 3套方案
        daily_used_ids = []  # 当日所有使用的菜品ID

        # 为每套方案生成早午晚三餐
        for plan_type in ["A", "B", "C"]:
            plan_meals = []
            plan_used_ids = []
            for meal_nut in meal_nutrition_list:
                recommended_meal, meal_used_ids = self._recommend_single_meal(
                    meal_nut, used_ids + daily_used_ids + plan_used_ids, plan_type
                )
                plan_meals.append(recommended_meal)
                plan_used_ids.extend(meal_used_ids)
            daily_plans.append(plan_meals)
            daily_used_ids.extend(plan_used_ids)

        return DailyDietPlan(
            day=1,  # 后续外部赋值具体天数
            daily_nutrition=daily_nutrition,
            plans=daily_plans,
            used_dish_ids=daily_used_ids
        )

    def recommend_multi_day_plan(self, info: PregnancyInfo, days: int = 7) -> WeeklyDietPlan:
        """
        生成多日饮食计划（默认7天）
        days: 生成天数（≥1）
        """
        if days < 1:
            raise ValueError("生成天数必须≥1")

        daily_plans = []
        all_used_ids = []  # 全局已使用菜品ID（避免跨天重复）

        for day in range(1, days + 1):
            # 生成当日计划（传入全局已使用ID）
            daily_plan = self.recommend_daily_plan(info, all_used_ids)
            daily_plan.day = day
            daily_plans.append(daily_plan)
            all_used_ids.extend(daily_plan.used_dish_ids)

            # 若菜品库用尽，提前终止
            if len(all_used_ids) >= len(self.all_dish_ids):
                print(f"提示：菜品库已用尽，实际生成{day}天计划")
                break

        return WeeklyDietPlan(
            info=info,
            daily_plans=daily_plans
        )

    def print_weekly_recommendation(self, weekly_plan: WeeklyDietPlan):
        """打印一周推荐结果（含每日3套方案）"""
        info = weekly_plan.pregnancy_info
        total_days = weekly_plan.total_days
        sample_nutrition = weekly_plan.daily_plans[0].daily_nutrition

        # 打印头部信息
        print("=" * 120)
        print(f"孕妇一周饮食推荐报告（{total_days}天）")
        print(
            f"【孕妇基础信息】BMI：{info.bmi} | 孕期：{info.trimester}期 | 孕前体重：{info.pre_pregnancy_weight}kg | 活动系数：{info.activity_level}")
        print(
            f"【每日营养目标】总能量：{sample_nutrition.total_energy}kcal | 蛋白质：{sample_nutrition.protein_amount}g | 碳水：{sample_nutrition.min_carb}-{sample_nutrition.max_carb}g | 脂肪：{sample_nutrition.min_fat}-{sample_nutrition.max_fat}g")
        print(f"【三餐能量配比】早餐40% | 午餐35% | 晚餐25%")
        print("=" * 120)

        # 打印每日计划
        for daily_plan in weekly_plan.daily_plans:
            print(f"\n【第{daily_plan.day}天】")
            # 打印3套方案
            plan_names = ["A方案（经典家常）", "B方案（清淡快手）", "C方案（高纤营养）"]
            for idx, (plan, plan_name) in enumerate(zip(daily_plan.plans, plan_names)):
                print(f"\n{plan_name}：")
                for meal in plan:
                    print(f"  {meal.meal_name}：")
                    print(
                        f"    营养对比：能量{meal.total_energy}kcal（目标{meal.target_energy}kcal）| 蛋白{meal.total_protein}g（目标{meal.target_protein}g）| 脂肪{meal.total_fat}g（目标{meal.target_fat}g）| 碳水{meal.total_carbohydrate}g（目标{meal.target_carbohydrate}g）")
                    print(f"    菜品组成（共{len(meal.servings)}道）：")
                    for serving in meal.servings:
                        print(
                            f"      - {serving.dish.name}：{serving.grams}克 | 食材：{serving.dish.composition} | 做法：{serving.dish.cooking_method}")
                        print(
                            f"        营养：{serving.energy}kcal | 蛋白{serving.protein}g | 脂肪{serving.fat}g | 碳水{serving.carbohydrate}g")
            print("\n" + "-" * 120)

    def export_weekly_to_json(self, weekly_plan: WeeklyDietPlan, save_path: Optional[str] = None) -> str:
        """
        导出一周计划为JSON格式
        :param weekly_plan: 一周饮食计划对象
        :param save_path: 可选，JSON文件保存路径（如"./孕妇一周饮食计划.json"）
        :return: JSON字符串
        """
        # 生成JSON字符串
        weekly_dict = weekly_diet_plan_to_json(weekly_plan)
        json_str = json.dumps(
            weekly_dict,
            ensure_ascii=False,  # 中文不转义
            indent=2,  # 格式化缩进
            sort_keys=False  # 不排序键
        )
        # 若指定保存路径，写入文件
        if save_path:
            try:
                with open(save_path, "w", encoding="utf-8") as f:
                    f.write(json_str)
                print(f"JSON文件已保存至：{save_path}")
            except Exception as e:
                print(f"保存JSON文件失败：{str(e)}")
        return json_str


def meal_nutrition_to_dict(meal_nut: MealNutrition) -> dict:
    """将MealNutrition类转为字典"""
    return {
        "meal_name": meal_nut.meal_name,
        "target_energy_kcal": meal_nut.energy,
        "target_protein_g": meal_nut.protein,
        "target_fat_g": meal_nut.fat,
        "target_carbohydrate_g": meal_nut.carbohydrate
    }


def serving_to_dict(serving: Serving) -> dict:
    """将Serving类转为字典"""
    return {
        "dish_id": serving.dish.dish_id,
        "dish_name": serving.dish.name,
        "category": serving.dish.core_category,
        "composition": serving.dish.composition,
        "serving_grams": serving.grams,
        "cooking_method": serving.dish.cooking_method,
        "icon": serving.dish.icon,
        "nutrition": {
            "energy_kcal": round(serving.energy),
            "protein_g": round(serving.protein, 1),
            "fat_g": round(serving.fat, 1),
            "carbohydrate_g": round(serving.carbohydrate, 1)
        }
    }


def recommended_meal_to_dict(meal: RecommendedMeal) -> dict:
    """将RecommendedMeal类转为字典"""
    return {
        "meal_name": meal.meal_name,
        "actual_nutrition": {
            "total_energy_kcal": meal.total_energy,
            "total_protein_g": meal.total_protein,
            "total_fat_g": meal.total_fat,
            "total_carbohydrate_g": meal.total_carbohydrate
        },
        "target_nutrition": {
            "target_energy_kcal": meal.target_energy,
            "target_protein_g": meal.target_protein,
            "target_fat_g": meal.target_fat,
            "target_carbohydrate_g": meal.target_carbohydrate
        },
        "dishes": [serving_to_dict(s) for s in meal.servings]  # 嵌套菜品列表
    }


def daily_diet_plan_to_dict(daily_plan: DailyDietPlan) -> dict:
    """将DailyDietPlan类转为字典（含3套方案）"""
    # 定义方案名称映射（A/B/C → 中文名称）
    plan_names = ["经典家常", "清淡快手", "高纤营养"]
    daily_dict = {
        "day": daily_plan.day,
        "daily_nutrition_target": {
            "total_energy_kcal": daily_plan.daily_nutrition.total_energy,
            "protein_g": [daily_plan.daily_nutrition.protein_amount],
            "carbohydrate_g_range": [
                daily_plan.daily_nutrition.min_carb,
                daily_plan.daily_nutrition.max_carb
            ],
            "fat_g_range": [
                daily_plan.daily_nutrition.min_fat,
                daily_plan.daily_nutrition.max_fat
            ],
            "energy_percentage_rule": {
                "breakfast": "40%",
                "lunch": "40%",
                "dinner": "20%"
            }
        },
        "daily_plans": {}  # 存储3套方案
    }
    # 填充3套方案（A/B/C）
    for idx, (plan, plan_name) in enumerate(zip(daily_plan.plans, plan_names)):
        meals = [recommended_meal_to_dict(meal) for meal in plan]
        daily_dict["daily_plans"][f"plan_{chr(65 + idx)}"] = {  # plan_A/plan_B/plan_C
            "plan_name": plan_name,
            "meals": meals,  # 早午晚三餐
            "total_actual_nutrition": {
                "total_energy_kcal": sum(dic['actual_nutrition']['total_energy_kcal'] for dic in meals),
                "total_protein_g": sum(dic['actual_nutrition']['total_protein_g'] for dic in meals),
                "total_fat_g": sum(dic['actual_nutrition']['total_fat_g'] for dic in meals),
                "total_carbohydrate_g": sum(dic['actual_nutrition']['total_carbohydrate_g'] for dic in meals)
            },
            "total_target_nutrition": {
                "target_energy_kcal": sum(dic['target_nutrition']['target_energy_kcal'] for dic in meals),
                "target_protein_g": sum(dic['target_nutrition']['target_protein_g'] for dic in meals),
                "target_fat_g": sum(dic['target_nutrition']['target_fat_g'] for dic in meals),
                "target_carbohydrate_g": sum(dic['target_nutrition']['target_carbohydrate_g'] for dic in meals)
            }
        }
    return daily_dict


def weekly_diet_plan_to_json(weekly_plan: WeeklyDietPlan) -> dict:
    """将WeeklyDietPlan类转为JSON字符串"""
    # 构建完整JSON结构
    weekly_dict = {
        "pregnancy_info": {
            "pre_pregnancy_bmi": weekly_plan.pregnancy_info.bmi,
            "trimester": weekly_plan.pregnancy_info.trimester,
            "pre_pregnancy_weight_kg": weekly_plan.pregnancy_info.pre_pregnancy_weight,
            "activity_level": weekly_plan.pregnancy_info.activity_level,
            "trimester_desc": {1: "孕早期", 2: "孕中期", 3: "孕晚期"}[weekly_plan.pregnancy_info.trimester]
        },
        "total_days": weekly_plan.total_days,
        "daily_plans": [daily_diet_plan_to_dict(day_plan) for day_plan in weekly_plan.daily_plans]
    }
    # 转为JSON字符串（确保中文不转义、数值保留1位小数）
    return weekly_dict


# 数据解析函数（适配Excel文件）
def diet_parsing(df: DataFrame) -> List[Dish]:
    """从Excel读取菜品数据（对应my_h_dish_nutrition_info_new.xlsx）"""
    dishes = []
    for _, row in df.iterrows():
        # 核心修改：跳过core_category为"其他"的菜品
        if str(row['core_category']).strip() == "其他":
            continue  # 直接忽略该菜品
        # 处理空值
        cooking_method = row['cooking_method'] if pd.notna(row['cooking_method']) else "家常做法"
        serving_size = row['serving_size'] if pd.notna(row['serving_size']) else "100"
        icon = row['icon'] if pd.notna(row['icon']) else ""
        # 构建Dish对象
        dish = Dish(
            dish_id=int(row['dish_id']),
            name=str(row['dish_name']),
            category=str(row['category']),
            core_category=str(row['core_category']),
            composition=str(row['composition']),
            serving_size=serving_size,
            energy=float(row['energy']) if pd.notna(row['energy']) else 0.0,
            protein=float(row['protein']) if pd.notna(row['protein']) else 0.0,
            fat=float(row['fat']) if pd.notna(row['fat']) else 0.0,
            carbohydrate=float(row['carbohydrates']) if pd.notna(row['carbohydrates']) else 0.0,
            cooking_method=cooking_method,
            recom_meal=str(row['recom_meal']),
            icon=icon
        )
        dishes.append(dish)
    return dishes


# 主函数（示例调用）
if __name__ == '__main__':
    # 1. 读取Excel菜品数据
    excel_path = "../../get_nutrition_data/my_h_dish_classify_meal.xlsx"  # 请确保路径正确
    try:
        df = pd.read_excel(excel_path)
        dishes = diet_parsing(df)
        print(f"成功加载{len(dishes)}道菜品（ID范围：{dishes[0].dish_id}-{dishes[-1].dish_id}）")
    except Exception as e:
        print(f"读取Excel失败：{str(e)}")
        exit()

    # 2. 初始化推荐器
    recommender = PregnancyDietRecommender(dishes)

    # 3. 输入孕妇信息（示例：BMI22.5，孕中期，孕前60kg，轻度活动）
    pregnant_info = PregnancyInfo(
        bmi=22.5,
        week=22,
        trimester=2,
        pre_pregnancy_weight=60.0,
        pregnancy_weight=65.0,
        activity_level=1.375
    )

    # 4. 生成7天计划（可修改days参数调整天数）
    weekly_plan = recommender.recommend_multi_day_plan(pregnant_info, days=1)

    # 5. 打印推荐结果
    recommender.print_weekly_recommendation(weekly_plan)

    # 6. 新增：导出JSON格式（两种方式可选）
    # 方式1：仅打印JSON字符串
    json_result = recommender.export_weekly_to_json(weekly_plan)
    print("\n【孕妇一周饮食计划 - JSON格式】")
    print(json_result)

    # 方式2：保存为JSON文件（推荐，便于后续使用）
    # json_result = recommender.export_weekly_to_json(
    #     weekly_plan,
    #     save_path="./孕妇一周饮食推荐计划.json"  # 自定义保存路径
    # )
