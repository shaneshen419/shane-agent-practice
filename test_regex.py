import re

task_description = "帮我安排一下北京国庆七天旅游计划"
print(f"任务描述: {task_description}")

# 尝试提取目的地 - 更精确的模式
# 匹配"安排一下"后面的地点名称
destination_match = re.search(r'安排一下([\u4e00-\u9fa5]{2,5})(?=国庆)', task_description)
destination = destination_match.group(1) if destination_match else None
print(f"目的地匹配结果: {destination}")

# 尝试提取天数 - 匹配中文数字或阿拉伯数字
days_match = re.search(r'(?:(\d+)|([一二三四五六七八九十]))天', task_description)
days = days_match.group(1) if days_match else None
# 如果匹配到中文数字，需要转换为阿拉伯数字
days_dict = {'一': '1', '二': '2', '三': '3', '四': '4', '五': '5', '六': '6', '七': '7', '八': '8', '九': '9', '十': '10'}
if not days and days_match and days_match.group(2):
    days = days_dict.get(days_match.group(2))
print(f"天数匹配结果: {days}")

# 尝试另一种模式
# 更宽松的匹配方式
destination_match2 = re.search(r'([\u4e00-\u9fa5]{2,5})(?=国庆|\d+天)', task_description)
destination2 = destination_match2.group(1) if destination_match2 else None
print(f"目的地匹配结果2: {destination2}")

# 匹配天数
days_match2 = re.search(r'([一二三四五六七八九十])天', task_description)
days2 = days_match2.group(1) if days_match2 else None
if days2:
    days2 = days_dict.get(days2)
print(f"天数匹配结果2: {days2}")