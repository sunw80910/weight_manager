import pandas as pd
import random
from typing import List, Dict, Tuple, Optional


class Exercise:
    """运动类，存储运动的相关信息"""

    def __init__(self, name: str, intensity: str, intensity_met: float, trimester_availability: List[bool],
                 icon: str, icon_type: str, exercise_category: str):
        """
        初始化运动对象

        参数:
            name: 运动名称
            intensity: 强度分类（低、中、高）
            intensity_met: 强度 MET 值
            trimester_availability: 各孕期是否适合的列表 [孕早期, 孕中期, 孕晚期]
            icon: icon地址
            icon_type: icon_type
            exercise_category: 运动分类
        """
        self.name = name
        self.intensity = intensity
        self.intensity_met = intensity_met
        self.trimester_availability = trimester_availability
        self.icon = icon
        self.icon_type = icon_type
        self.exercise_category = exercise_category

    def get_calories(self, duration_minutes: int, weight_kg: float = 60) -> float:
        """
        计算指定时长和体重下消耗的热量
        参数:
            duration_minutes: 运动时长（分钟）
            weight_kg: 体重（kg），默认60kg
        返回:
            消耗的热量（千卡）
        """
        # 根据体重调整热量消耗
        return round(self.intensity_met * weight_kg * (duration_minutes / 60.0), 1)


class ExercisePlanner:
    """运动计划生成器"""

    def __init__(self, exercises_df: pd.DataFrame):
        """
        初始化运动计划生成器
        参数:
            exercises_df: 包含运动数据的DataFrame
        """
        self.exercises = self._parse_exercises(exercises_df)
        self.days_of_week = []
        self.user_preferences = []  # 用户偏好运动列表
        self.max_daily_duration = 60  # 每日最大运动时间（分钟）
        self.min_exercise_duration = 10  # 单个运动最小持续时间（分钟）
        self.max_single_exercise_duration = 30  # 单个运动最大持续时间（分钟）
        self.excluded_exercises = set()  # 当天需要排除的运动
        self.exercise_history = []  # 运动历史记录
        self._validate_duration()

    def set_max_daily_duration(self, minutes: int):
        """设置每日最大运动时间"""
        self.max_daily_duration = minutes

    def set_max_single_exercise_duration(self, minutes: int):
        """设置单个运动的最大持续时间"""
        self.max_single_exercise_duration = minutes
        self._validate_duration()

    def _validate_duration(self):
        if self.min_exercise_duration >= self.max_single_exercise_duration:
            raise ValueError("单个运动最小持续时间必须小于单个运动最大持续时间")

    def set_user_preferences(self, preferred_exercises: List[str], trimester: int):
        """
                设置用户偏好的运动

                参数:
                    preferred_exercises: 用户偏好的运动名称列表
                    trimester: 孕妇孕期
                """
        # 过滤出实际存在且适合当前孕期的偏好运动
        self.user_preferences = [
            exercise for exercise in self.exercises
            if exercise.exercise_category in preferred_exercises
               and exercise.trimester_availability[trimester - 1]  # 校验当前孕期是否可做
        ]

    def _parse_exercises(self, df: pd.DataFrame) -> List[Exercise]:
        """
        从DataFrame解析运动数据

        参数:
            df: 包含运动数据的DataFrame

        返回:
            运动对象列表
        """
        exercises = []
        for _, row in df.iterrows():
            exercise = Exercise(
                name=row['activity'],
                intensity=row['intensity'],
                intensity_met=row['MET'],
                trimester_availability=[bool(row['tri_1st_ok']),
                                        bool(row['tri_2nd_ok']),
                                        bool(row['tri_3rd_ok'])],
                icon=row['icon'],
                icon_type=row['icon_type'],
                exercise_category=row['exercise_category']
            )
            exercises.append(exercise)
        return exercises

    def _select_exercise(self, trimester: int, target_calories: float, low_intensity: list[Exercise],
                         moderate_intensity: list[Exercise], weight_kg: float) -> list[tuple[Exercise, int]]:
        # 根据孕期确定基础运动池
        if trimester == 1:
            primary_pool = low_intensity
            secondary_pool = moderate_intensity
            if not primary_pool:
                primary_pool = moderate_intensity
        elif trimester == 2:
            primary_pool = moderate_intensity
            secondary_pool = low_intensity
            if not primary_pool:
                primary_pool = low_intensity
        else:
            primary_pool = low_intensity + [e for e in moderate_intensity if
                                            e.exercise_category not in ["游泳", "固定自行车"]]
            secondary_pool = [e for e in moderate_intensity if e.exercise_category in ["孕妇瑜伽", "拉伸"]]
            if not primary_pool:
                primary_pool = secondary_pool if secondary_pool else low_intensity + moderate_intensity

        exercise_pool = primary_pool
        preferred_available = [e for e in self.user_preferences if e in exercise_pool or e in secondary_pool]

        # 计算基于体重的每分钟卡路里消耗
        def get_calories_per_min(exercise: Exercise) -> float:
            return exercise.get_calories(1, weight_kg)

        # 严格遵守每日运动时间上限，不临时放宽
        target_duration = self.max_daily_duration

        # 能量缺口完成度追踪
        current_calories = 0
        exercise_sessions = []
        remaining_duration = target_duration
        remaining_calories = target_calories

        # 按效率排序的运动池
        all_available = sorted(
            low_intensity + moderate_intensity,
            key=lambda e: e.intensity_met,
            reverse=True
        )

        # 计算平均每分钟消耗的卡路里
        if all_available:
            avg_cal_per_min = sum(e.intensity_met * weight_kg / 60 for e in all_available) / len(all_available)
        else:
            avg_cal_per_min = 0

        # 定义能量缺口类型
        max_possible_daily_cal = target_duration * avg_cal_per_min
        is_small_deficit = target_calories < max_possible_daily_cal * 0.3  # 小缺口：<30%最大可能消耗
        is_moderate_deficit = (target_calories >= max_possible_daily_cal * 0.3 and
                               target_calories <= max_possible_daily_cal * 0.7)  # 适中缺口：30%-70%
        is_large_deficit = target_calories > max_possible_daily_cal * 0.7  # 大缺口：>70%

        # 循环选择运动，直到达到时间上限或热量目标
        loop_counter = 0
        max_loops = 20  # 增加最大循环次数，确保有足够尝试机会
        while remaining_duration > 0 and remaining_calories > 0 and loop_counter < max_loops:
            # 筛选可用运动（排除最近使用过的）
            recently_used = [ex.exercise_category for ex, _ in exercise_sessions]
            available_pool = [e for e in all_available if e.exercise_category not in recently_used]

            # 如果没有可用运动，重置限制
            if not available_pool:
                available_pool = all_available

            # 小能量缺口策略：优先低强度运动，精确控制时长
            if is_small_deficit:
                # 1. 优先选择用户偏好的低强度运动
                preferred_low = [e for e in available_pool
                                 if e in preferred_available and e.intensity == "低"]
                if preferred_low:
                    exercise = random.choice(preferred_low)
                else:
                    # 2. 没有偏好则选择低强度运动
                    low_pool = [e for e in available_pool if e.intensity == "低"]
                    if low_pool:
                        exercise = random.choice(low_pool)
                    else:
                        # 3. 迫不得已才选高强度，但严格控制时长
                        exercise = min(available_pool, key=lambda e: e.intensity_met)  # 选强度最低的
            # 当能量缺口适中时，更优先选择偏好运动
            elif is_moderate_deficit:
                # 70%概率优先选择偏好运动
                if preferred_available and random.random() < 0.7:
                    # 从偏好池中选择最高效的
                    preferred_pool = [e for e in available_pool if e in preferred_available]
                    if preferred_pool:
                        exercise = max(preferred_pool, key=lambda e: e.intensity_met)
                    else:
                        exercise = max(available_pool, key=lambda e: e.intensity_met)
                else:
                    # 30%概率按效率随机选择
                    weights = [e.intensity_met for e in available_pool]
                    exercise = random.choices(available_pool, weights=weights, k=1)[0]
            else:
                # # 大能量缺口策略：优先高效运动  is_large_deficit
                # 动态调整选择策略：当剩余热量高时，优先高效运动
                heat_ratio = remaining_calories / target_calories
                time_ratio = remaining_duration / target_duration

                # 热量缺口大且时间充足时，强制选择高效运动
                if heat_ratio > 0.6 and time_ratio > 0.5:
                    exercise = max(available_pool, key=lambda e: e.intensity_met)
                else:
                    # 确保至少包含一些用户偏好的运动
                    has_preferred = any(e in preferred_available for e in [ex for ex, _ in exercise_sessions])

                    # 调整偏好运动的选择策略
                    if not has_preferred and preferred_available:
                        # 从偏好池中随机选择
                        preferred_pool = [e for e in available_pool if e in preferred_available]
                        if preferred_pool:
                            exercise = random.choice(preferred_pool)
                        else:
                            exercise = max(available_pool, key=lambda e: e.intensity_met)
                    else:
                        # 基于效率的随机选择
                        weights = [e.intensity_met for e in available_pool]
                        exercise = random.choices(available_pool, weights=weights, k=1)[0]

            # 计算该运动每分钟消耗的热量
            cal_per_min = get_calories_per_min(exercise)

            # 小缺口时精确控制时长（避免过量）
            if is_small_deficit:
                # 计算刚好满足剩余热量的时长（向下取整）
                required_duration = remaining_calories / cal_per_min
                # 最多分配剩余时间的80%，留有余地
                session_duration = min(required_duration * 0.8, remaining_duration)
                # 确保不低于最小时长
                session_duration = max(session_duration, self.min_exercise_duration)
            else:
                # 计算可分配的时长（考虑最小/最大限制）
                max_possible_duration = min(remaining_duration, self.max_single_exercise_duration)
                # 计算需要的时长（基于剩余热量）
                required_duration_for_calories = remaining_calories / cal_per_min
                # 确定实际分配时长
                session_duration = min(max_possible_duration, required_duration_for_calories,
                                       self.max_single_exercise_duration)
                # 确保最小持续时间
                session_duration = max(session_duration, self.min_exercise_duration)

            # 关键改进：当剩余时间不足以分配最小单位时，尝试拆分或退出
            if session_duration > remaining_duration:
                # 如果剩余时间连最小运动都无法完成，尝试拆分高效运动
                if remaining_duration >= self.min_exercise_duration * 0.5:
                    session_duration = remaining_duration
                else:
                    # 剩余时间太少，直接退出
                    break

            # 应用时长取整（保留5的倍数）
            session_duration = int(round(session_duration / 5) * 5)
            # 如果取整后时长为0，跳过
            if session_duration == 0:
                break

            # 添加运动时段
            exercise_sessions.append((exercise, session_duration))
            # 更新剩余时间和热量
            remaining_duration -= session_duration
            remaining_calories -= cal_per_min * session_duration
            current_calories += cal_per_min * session_duration
            loop_counter += 1

        # 小缺口时不强制补足（避免过量），大缺口时才补足
        if remaining_calories > 0 and remaining_duration > 0 and not is_small_deficit:
            required_cal_per_min = remaining_calories / remaining_duration
            efficient_pool = [e for e in all_available if get_calories_per_min(e) >= required_cal_per_min]

            if efficient_pool:
                exercise = max(efficient_pool, key=lambda e: e.intensity_met)
                cal_per_min = get_calories_per_min(exercise)
                session_duration = remaining_duration
                exercise_sessions.append((exercise, session_duration))
                remaining_calories -= cal_per_min * session_duration
            else:
                exercise = max(all_available, key=lambda e: e.intensity_met)
                cal_per_min = get_calories_per_min(exercise)
                session_duration = remaining_duration
                exercise_sessions.append((exercise, session_duration))
                remaining_calories -= cal_per_min * session_duration

        return exercise_sessions

    def _merge_exercises(self, sessions: List[Tuple[Exercise, int]]) -> List[Tuple[Exercise, int]]:
        """合并重复的运动项目，确保同一类型运动总时长不超过限制"""
        if not sessions:
            return []

        merged = []

        for exercise, duration in sessions:
            # 先检查当前单次时长是否超限
            if duration > self.max_single_exercise_duration:
                # 若单次时长超限，拆分为多个时段
                while duration > 0:
                    split_duration = min(duration, self.max_single_exercise_duration)
                    merged.append((exercise, split_duration))
                    duration -= split_duration
                continue  # 跳过后续合并逻辑
            # 如果是相同类型的运动
            last_idx = len(merged) - 1
            if last_idx >= 0 and merged[last_idx][0].name == exercise.name:
                # 计算合并后的总时长
                new_total = merged[last_idx][1] + duration

                # 如果合并后不超过最大时长，合并
                if new_total <= self.max_single_exercise_duration:
                    merged[last_idx] = (merged[last_idx][0], new_total)
                else:
                    # 否则，作为新的运动时段添加
                    merged.append((exercise, duration))
            else:
                # 不同类型的运动，直接添加
                merged.append((exercise, duration))

        return merged

    def _update_excluded_exercises(self):
        """更新当天需要排除的运动类型"""
        self.excluded_exercises = set()

        if self.exercise_history and self.exercise_history[-1]:
            # 修正：排除上一次运动时，保留偏好运动（不排除偏好）
            last_exercises = self.exercise_history[-1]
            self.excluded_exercises.update([
                ex for ex in last_exercises
                if ex not in [pref.exercise_category for pref in self.user_preferences]  # 关键：偏好运动不被排除
            ])

        # 30%概率额外排除非偏好运动
        if random.random() < 0.3 and self.exercise_history:
            all_history_exercises = [ex for day_ex in self.exercise_history if day_ex for ex in day_ex]
            non_preferred_history = [
                ex for ex in all_history_exercises
                if ex not in [pref.exercise_category for pref in self.user_preferences]
            ]
            if non_preferred_history:
                self.excluded_exercises.add(random.choice(non_preferred_history))

    def generate_weekly_plan(self, trimester: int, weekly_calorie_deficit: float, weight_kg: float,
                             rest_days: List[str], planned_days=7, preferred_exercises: List[str] = None) -> Dict:
        """生成每周运动计划，考虑每日运动时间上限"""
        if trimester not in [1, 2, 3]:
            raise ValueError("孕期必须是1（孕早期）、2（孕中期）或3（孕晚期）")

        self.days_of_week = [f'day_{i}' for i in range(1, planned_days + 1)]

        # 新增：检查能量缺口是否为正数
        lowest = round(1.5 * weight_kg / 6 * planned_days, 1)
        weekly_calorie_deficit = lowest if weekly_calorie_deficit * planned_days <= lowest else weekly_calorie_deficit * planned_days

        # 设置用户偏好
        if preferred_exercises:
            self.set_user_preferences(preferred_exercises, trimester)

        # 重置历史记录
        self.exercise_history = []

        # 可用的运动列表
        available_exercises = [e for e in self.exercises if e.trimester_availability[trimester - 1]]

        # 按强度分类运动
        low_intensity = [e for e in available_exercises if e.intensity == "低"]
        moderate_intensity = [e for e in available_exercises if e.intensity == "中"]

        # 计算每天需要消耗的平均热量
        rest_days = rest_days if rest_days else []
        active_days = [day for day in self.days_of_week if day not in rest_days]
        avg_daily_calories = round(weekly_calorie_deficit / len(active_days) if active_days else 0, 1)
        # if not active_days:
        #     raise ValueError("活跃天数不能为0")
        # avg_daily_calories = weekly_calorie_deficit / len(active_days)  # 确保除法正确

        # 为每天分配运动
        weekly_plan = {}
        week_total_calories = 0
        for day in self.days_of_week:
            if day in rest_days:
                weekly_plan[day] = {"exercises": [{"name": "休息", "duration": 0, "calories": 0}]}
                self.exercise_history.append(None)  # 休息日记录为None
            else:
                # 获取当天应排除的运动类型
                self._update_excluded_exercises()

                # 选择运动组合（排除指定类型）
                filtered_low = [e for e in low_intensity if e.exercise_category not in self.excluded_exercises]
                filtered_moderate = [e for e in moderate_intensity if
                                     e.exercise_category not in self.excluded_exercises]

                # 如果过滤后没有可用运动，放宽限制
                if not filtered_low and not filtered_moderate:
                    filtered_low = low_intensity
                    filtered_moderate = moderate_intensity

                exercise_sessions = self._select_exercise(trimester, avg_daily_calories, filtered_low,
                                                          filtered_moderate, weight_kg)

                # 合并重复的运动项目
                merged_sessions = self._merge_exercises(exercise_sessions)
                # merged_sessions = exercise_sessions

                # 计算总消耗
                show_total_calories = round(sum(ex.get_calories(dur, weight_kg) for ex, dur in merged_sessions), 1)
                total_calories = round(sum(ex.get_calories(dur, 55) for ex, dur in merged_sessions), 1)
                # 记录当天运动名称（用于排除逻辑）
                self.exercise_history.append([ex.name for ex, _ in merged_sessions])

                weekly_plan[day] = {
                    "exercises": [
                        {"name": ex.name, "icon": ex.icon, "icon_type": ex.icon_type, "duration": dur,
                         "show_calories": round(ex.get_calories(dur, weight_kg)),
                         "calories": round(ex.get_calories(dur, 55))}
                        for ex, dur in merged_sessions
                    ],
                    "total_duration": sum(dur for _, dur in merged_sessions),
                    "total_calories": round(total_calories),
                    "show_total_calories": round(show_total_calories),
                    "target_calories": round(avg_daily_calories),  # 记录每日目标热量
                    "energy_gap_completion": round(total_calories / avg_daily_calories, 4)  # 能量缺口完成度
                }
                week_total_calories += total_calories if total_calories else 0

        #  能量缺口完成度
        weekly_plan["week_total_calories"] = round(week_total_calories)
        weekly_plan["weekly_calorie_deficits"] = weekly_calorie_deficit
        weekly_plan["energy_gap_completio"] = round(week_total_calories / weekly_calorie_deficit, 4)
        weekly_plan["user_preferences"] = preferred_exercises
        return weekly_plan

    def print_weekly_plan(self, plan: Dict, trimester: int, weekly_calorie_deficit: float, weight_kg: float):
        """打印每周运动计划，支持多项运动"""
        trimester_names = ["孕早期", "孕中期", "孕晚期"]
        print(f"\n{'=' * 40}")
        print(f"{trimester_names[trimester - 1]}({trimester}期)运动周计划")
        print(f"目标：每周制造 {weekly_calorie_deficit:.1f} 千卡能量缺口")
        print(f"体重：{weight_kg} kg")
        print(f"每日最大运动时间：{self.max_daily_duration} 分钟")
        print(f"用户偏好运动：{[pref.name for pref in self.user_preferences] if self.user_preferences else '无'}")
        print(f"{'=' * 40}")

        total_calories = 0
        for day, info in plan.items():
            if day not in self.days_of_week:
                continue
            if info['exercises'][0]['name'] == "休息":
                print(f"{day}: 休息")
            else:
                print(f"{day}（目标：{info['target_calories']:.1f}千卡）:")
                for exercise in info['exercises']:
                    print(f"  - {exercise['name']} ({exercise['duration']}分钟) - 消耗 {exercise['calories']:.1f} 千卡")
                    print(f"  - {exercise['name']} ({exercise['duration']}分钟) - 真消耗 {exercise['show_calories']:.1f} 千卡")
                print(f"  当日总计: {info['total_duration']}分钟 | 实际消耗: {info['total_calories']:.1f}千卡 "
                      f"| 完成度: {info['total_calories'] / info['target_calories'] * 100:.1f}%")
                print(f"  当日总计: {info['total_duration']}分钟 | 真实际消耗: {info['show_total_calories']:.1f}千卡 "
                      f"| 真完成度: {info['show_total_calories'] / info['target_calories'] * 100:.1f}%")
            total_calories += info['total_calories'] if 'total_calories' in info else 0

        print(f"{'=' * 40}")
        print(
            f"总计：预计消耗 {plan.get('week_total_calories'):.1f} 千卡 | 目标缺口：{plan.get("weekly_calorie_deficits"):.1f} 千卡")
        print(f"能量缺口完成度：{plan.get("energy_gap_completio")}")
        print(f"{'=' * 40}")


def get_pregnancy(week):
    if week <= 12:
        return '孕早期'
    elif week < 28:
        return '孕中期'
    else:
        return '孕晚期'


if __name__ == "__main__":
    try:
        df = pd.read_excel('../../data/运动强度.xlsx')
    except FileNotFoundError:
        print("找不到运动数据文件，请确保文件路径正确。")

    # 创建运动计划生成器
    planner = ExercisePlanner(df)

    #  孕周
    week = 35
    pregnancy = get_pregnancy(week)

    trimesters = {
        "孕早期": 1,
        "孕中期": 2,
        "孕晚期": 3
    }
    # 示例：为不同孕期生成运动计划 [1,2,3]
    trimester = trimesters.get(pregnancy)
    weekly_calorie_deficits = 150
    weight_kg = 80
    # 用户偏好的运动列表
    user_preferences = ["散步", "孕妇瑜伽", "拉伸"]
    # 休息日
    rest_days = []

    plan = planner.generate_weekly_plan(
        trimester=trimester,
        weekly_calorie_deficit=weekly_calorie_deficits,
        weight_kg=weight_kg,
        rest_days=rest_days,
        planned_days=7,
        preferred_exercises=user_preferences
    )
    print(plan)
    planner.print_weekly_plan(plan, trimester, weekly_calorie_deficits, weight_kg)
