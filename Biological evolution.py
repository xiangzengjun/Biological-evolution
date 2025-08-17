import pygame
import pygame.freetype
import numpy as np
import random
import math
from enum import Enum
import matplotlib.pyplot as plt
from collections import defaultdict
import os
import sys

# 初始化pygame
pygame.init()
WIDTH, HEIGHT = 1000, 700
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("单细胞生物进化模拟")
clock = pygame.time.Clock()

# 环境参数
class Environment:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.food_grid = np.zeros((width, height))
        self.temperature = 25.0  # 摄氏度
        self.toxicity = 0.1  # 环境毒性
        self.food_multiplier = 1.0  # 食物生成倍数
        self.time = 0
        self.generate_food()
    
    def update(self):
        """更新环境状态"""
        self.time += 1
        # 温度缓慢波动变化
        self.temperature += random.uniform(-0.3, 0.3)
        self.temperature = max(5, min(35, self.temperature))
        
        # 毒性随时间变化
        self.toxicity += random.uniform(-0.02, 0.02)
        self.toxicity = max(0, min(0.5, self.toxicity))
        
        self.generate_food()
    
    def generate_food(self):
        """基于正弦函数的自然食物分布，随时间再生"""
        x_coords, y_coords = np.meshgrid(np.arange(self.width), np.arange(self.height), indexing='ij')
        
        # 使用正弦函数创建自然分布模式
        base_pattern = (np.sin(x_coords / 15.0) * np.cos(y_coords / 12.0) + 
                       np.sin(x_coords / 25.0 + self.time * 0.01) * np.sin(y_coords / 20.0))
        
        # 归一化到0-1范围
        base_pattern = (base_pattern + 2) / 4
        
        # 添加随机噪声和时间变化
        noise = np.random.uniform(0, 0.2, (self.width, self.height))
        time_factor = 0.8 + 0.2 * np.sin(self.time * 0.02)
        
        # 应用食物倍数控制
        self.food_grid = np.clip((base_pattern * time_factor + noise) * self.food_multiplier, 0, 1)

# 生物特征枚举
class Trait(Enum):
    SIZE = 1
    SPEED = 2
    SENSE = 3
    REPRODUCTION = 4
    DEFENSE = 5
    LIFESPAN = 6
    METABOLISM = 7
    TOXIN_RESISTANCE = 8

# 生物类
class Organism:
    ID_COUNTER = 0
    BASE_REPRODUCTION_COST = 60
    
    def __init__(self, x, y, dna=None, generation=1):
        self.id = Organism.ID_COUNTER
        Organism.ID_COUNTER += 1
        self.x = float(x)
        self.y = float(y)
        self.energy = 100.0
        self.age = 0
        self.generation = generation
        self.alive = True
        self.children = 0
        self.last_reproduction = 0
        
        if dna is None:
            # 初始DNA - 随机生成平衡的特征
            self.dna = {
                Trait.SIZE: random.uniform(0.2, 0.8),
                Trait.SPEED: random.uniform(0.2, 0.8),
                Trait.SENSE: random.uniform(0.2, 0.8),
                Trait.REPRODUCTION: random.uniform(0.2, 0.8),
                Trait.DEFENSE: random.uniform(0.2, 0.8),
                Trait.LIFESPAN: random.uniform(0.2, 0.8),
                Trait.METABOLISM: random.uniform(0.2, 0.8),
                Trait.TOXIN_RESISTANCE: random.uniform(0.2, 0.8)
            }
        else:
            self.dna = dna.copy()
            self.mutate()
        
        self.color = self.generate_color()
        self.max_lifespan = self.calculate_max_lifespan()
    
    def mutate(self):
        """DNA突变机制"""
        mutation_rate = 0.08  # 提高突变率促进进化
        mutation_strength = 0.15
        
        for trait in self.dna:
            if random.random() < mutation_rate:
                # 随机突变
                change = random.uniform(-mutation_strength, mutation_strength)
                self.dna[trait] = max(0.05, min(0.95, self.dna[trait] + change))
    
    def generate_color(self):
        """根据生物特征生成颜色"""
        # 使用不同特征组合生成更丰富的颜色
        r = int(100 + 155 * (self.dna[Trait.SPEED] * 0.6 + self.dna[Trait.SIZE] * 0.4))
        g = int(100 + 155 * (self.dna[Trait.DEFENSE] * 0.7 + self.dna[Trait.REPRODUCTION] * 0.3))
        b = int(100 + 155 * (self.dna[Trait.TOXIN_RESISTANCE] * 0.8 + self.dna[Trait.SENSE] * 0.2))
        return (r, g, b)
    
    def calculate_max_lifespan(self):
        """根据寿命基因计算最大寿命"""
        base_lifespan = 150
        lifespan_bonus = self.dna[Trait.LIFESPAN] * 300
        return int(base_lifespan + lifespan_bonus)
    
    def move(self, environment):
        """生物移动行为 - 根据感知寻找食物"""
        # 速度受SPEED基因影响
        max_speed = self.dna[Trait.SPEED] * 4.0
        # 感知范围受SENSE基因影响
        sense_range = int(self.dna[Trait.SENSE] * 25) + 5
        
        # 感知周围环境，寻找食物
        best_food = 0
        best_x, best_y = self.x, self.y
        
        # 在感知范围内搜索食物
        search_step = max(1, sense_range // 8)  # 优化搜索效率
        for dx in range(-sense_range, sense_range + 1, search_step):
            for dy in range(-sense_range, sense_range + 1, search_step):
                nx, ny = int(self.x + dx), int(self.y + dy)
                if 0 <= nx < environment.width and 0 <= ny < environment.height:
                    # 距离越远，感知能力越弱
                    distance = math.sqrt(dx*dx + dy*dy)
                    sense_efficiency = max(0.1, 1.0 - distance / sense_range)
                    food_val = environment.food_grid[nx][ny] * sense_efficiency
                    
                    if food_val > best_food:
                        best_food = food_val
                        best_x, best_y = nx, ny
        
        # 计算移动成本（受新陈代谢和体型影响）
        base_move_cost = 0.3 + self.dna[Trait.SIZE] * 0.2  # 体型越大消耗越多
        metabolism_factor = 0.5 + self.dna[Trait.METABOLISM] * 1.0
        
        # 向食物方向移动
        if best_food > 0.1:  # 只有足够的食物才值得移动
            dx = best_x - self.x
            dy = best_y - self.y
            dist = max(0.1, math.sqrt(dx*dx + dy*dy))
            
            # 计算实际移动距离
            move_distance = min(max_speed, dist)
            move_cost = base_move_cost * metabolism_factor * (move_distance / max_speed)
            
            if self.energy > move_cost:
                self.x += (dx / dist) * move_distance
                self.y += (dy / dist) * move_distance
                self.energy -= move_cost
        else:
            # 随机移动（探索行为）
            explore_distance = max_speed * 0.3
            angle = random.uniform(0, 2 * math.pi)
            self.x += math.cos(angle) * explore_distance
            self.y += math.sin(angle) * explore_distance
            self.energy -= base_move_cost * metabolism_factor * 0.3
        
        # 边界检查
        self.x = max(0, min(environment.width - 1, self.x))
        self.y = max(0, min(environment.height - 1, self.y))
    
    def eat(self, environment):
        """进食行为 - 消耗环境中的食物"""
        x_pos, y_pos = int(self.x), int(self.y)
        if 0 <= x_pos < environment.width and 0 <= y_pos < environment.height:
            food_val = environment.food_grid[x_pos][y_pos]
            if food_val > 0.05:  # 只有足够的食物才值得进食
                # 体型影响进食量 - 体型越大能吃得越多
                max_eat_amount = 0.08 + self.dna[Trait.SIZE] * 0.15
                eat_amount = min(food_val, max_eat_amount)
                
                # 消耗食物
                environment.food_grid[x_pos][y_pos] -= eat_amount
                
                # 转化为能量（效率受新陈代谢影响）
                energy_efficiency = 0.7 + self.dna[Trait.METABOLISM] * 0.3
                energy_gain = eat_amount * 45 * energy_efficiency
                self.energy += energy_gain
    
    def reproduce(self):
        """繁殖行为 - 消耗能量产生后代"""
        # 繁殖成本受繁殖基因影响
        base_cost = Organism.BASE_REPRODUCTION_COST
        reproduction_efficiency = self.dna[Trait.REPRODUCTION]
        reproduction_cost = base_cost * (1.5 - reproduction_efficiency * 0.5)
        
        # 繁殖条件检查
        min_age = 30
        min_energy = reproduction_cost + 20
        cooldown_period = int(40 * (1.1 - reproduction_efficiency))
        
        if (self.energy > min_energy and 
            self.age > min_age and 
            self.age - self.last_reproduction > cooldown_period):
            
            self.energy -= reproduction_cost
            self.children += 1
            self.last_reproduction = self.age
            
            # 创建后代，位置稍微偏移
            offspring_x = self.x + random.uniform(-8, 8)
            offspring_y = self.y + random.uniform(-8, 8)
            
            return Organism(
                offspring_x,
                offspring_y,
                self.dna,
                self.generation + 1
            )
        return None
    
    def update(self, environment):
        """更新生物状态"""
        if not self.alive:
            return None
        
        self.age += 1
        
        # 基础新陈代谢消耗
        base_metabolism = 0.4 + self.dna[Trait.METABOLISM] * 0.6
        self.energy -= base_metabolism
        
        # 环境温度影响（防御基因提供抗性）
        optimal_temp = 20.0
        temp_stress = abs(environment.temperature - optimal_temp) / 15.0
        defense_factor = self.dna[Trait.DEFENSE]
        temp_damage = temp_stress * (1.0 - defense_factor * 0.7) * 0.3
        self.energy -= temp_damage
        
        # 环境毒性影响（毒素抗性基因提供保护）
        toxin_resistance = self.dna[Trait.TOXIN_RESISTANCE]
        toxin_damage = max(0, environment.toxicity - toxin_resistance * 0.6) * 8
        self.energy -= toxin_damage
        
        # 年龄相关的衰老
        age_factor = self.age / self.max_lifespan
        if age_factor > 0.7:  # 进入老年期
            aging_damage = (age_factor - 0.7) * 2.0
            self.energy -= aging_damage
        
        # 死亡条件检查
        if self.age >= self.max_lifespan or self.energy <= 0:
            self.alive = False
            return None
        
        # 生物行为
        self.move(environment)
        self.eat(environment)
        
        # 尝试繁殖
        offspring = self.reproduce()
        return offspring

# 进化模拟类
class EvolutionSimulation:
    def __init__(self, width=200, height=150):
        self.organisms = []
        self.environment = Environment(width, height)
        self.width = width
        self.height = height
        self.time = 0
        self.population_history = []
        self.trait_history = defaultdict(list)
        self.generation_history = []
        self.deaths_by_cause = {'energy': 0, 'age': 0, 'environment': 0}
        
        # 创建初始生物种群
        initial_population = 50
        for _ in range(initial_population):
            x = random.randint(10, width - 10)
            y = random.randint(10, height - 10)
            self.organisms.append(Organism(x, y))
    
    def update(self):
        """更新模拟状态"""
        self.time += 1
        self.environment.update()
        
        surviving_organisms = []
        new_offspring = []
        deaths_this_turn = 0
        births_this_turn = 0
        
        # 更新所有生物
        for org in self.organisms:
            if org.alive:
                offspring = org.update(self.environment)
                
                if org.alive:
                    surviving_organisms.append(org)
                    if offspring:
                        new_offspring.append(offspring)
                        births_this_turn += 1
                else:
                    deaths_this_turn += 1
                    # 记录死亡原因
                    if org.energy <= 0:
                        if org.age >= org.max_lifespan:
                            self.deaths_by_cause['age'] += 1
                        else:
                            self.deaths_by_cause['energy'] += 1
                    else:
                        self.deaths_by_cause['environment'] += 1
        
        # 合并存活生物和新生儿
        self.organisms = surviving_organisms + new_offspring
        
        # 环境压力 - 如果种群过大，增加竞争压力
        max_population = 10000
        if len(self.organisms) > max_population:
            # 移除适应性最差的个体
            self.organisms.sort(key=lambda org: org.energy + org.age * 0.1, reverse=True)
            self.organisms = self.organisms[:max_population]
        
        # 如果种群过小，添加移民（模拟基因流）
        min_population = 8
        if len(self.organisms) < min_population:
            immigrants = min_population - len(self.organisms)
            for _ in range(immigrants):
                x = random.randint(5, self.width - 5)
                y = random.randint(5, self.height - 5)
                # 移民带来新的基因变异
                immigrant = Organism(x, y)
                # 额外突变以增加遗传多样性
                for trait in immigrant.dna:
                    if random.random() < 0.3:
                        change = random.uniform(-0.2, 0.2)
                        immigrant.dna[trait] = max(0.05, min(0.95, immigrant.dna[trait] + change))
                immigrant.color = immigrant.generate_color()
                self.organisms.append(immigrant)
        
        # 记录统计数据
        self.population_history.append(len(self.organisms))
        
        if self.organisms:
            # 记录特征平均值
            for trait in Trait:
                trait_avg = sum(org.dna[trait] for org in self.organisms) / len(self.organisms)
                self.trait_history[trait].append(trait_avg)
            
            # 记录世代信息
            avg_generation = sum(org.generation for org in self.organisms) / len(self.organisms)
            max_generation = max(org.generation for org in self.organisms)
            self.generation_history.append((avg_generation, max_generation))
        
        return deaths_this_turn, births_this_turn
    
    def draw(self, surface, scale_x, scale_y):
        """绘制模拟状态"""
        # 绘制食物分布背景（绿色背景表示食物分布，颜色越亮食物越多）
        food_surface = pygame.Surface((self.width, self.height))
        for x in range(self.width):
            for y in range(self.height):
                food_val = self.environment.food_grid[x][y]
                # 基础绿色背景
                base_green = 20
                food_green = int(base_green + food_val * 120)
                color = (0, min(255, food_green), 0)
                food_surface.set_at((x, y), color)
        
        # 缩放食物分布到屏幕尺寸
        scaled_food = pygame.transform.scale(food_surface, (int(self.width * scale_x), int(self.height * scale_y)))
        surface.blit(scaled_food, (0, 0))
        
        # 绘制生物（圆形生物体，颜色表示特征组合）
        for org in self.organisms:
            # 体型影响可见大小
            base_size = 4
            size_bonus = org.dna[Trait.SIZE] * 8
            size = int(base_size + size_bonus)
            
            pos_x = int(org.x * scale_x)
            pos_y = int(org.y * scale_y)
            
            # 绘制生物主体
            pygame.draw.circle(surface, org.color, (pos_x, pos_y), size)
            
            # 绘制生物边框（显示健康状态）
            health_ratio = org.energy / 100.0
            if health_ratio > 0.7:
                border_color = (0, 255, 0)  # 健康 - 绿色
            elif health_ratio > 0.3:
                border_color = (255, 255, 0)  # 一般 - 黄色
            else:
                border_color = (255, 0, 0)  # 危险 - 红色
            
            pygame.draw.circle(surface, border_color, (pos_x, pos_y), size, 1)
        
        # 绘制实时环境参数和种群信息
        # 使用更可靠的中文字体加载方法
        def get_chinese_font(size):
            # 尝试加载系统中文字体
            font_paths = [
                'C:/Windows/Fonts/msyh.ttc',  # 微软雅黑
                'C:/Windows/Fonts/simhei.ttf',  # 黑体
                'C:/Windows/Fonts/simsun.ttc',  # 宋体
                'C:/Windows/Fonts/simkai.ttf',  # 楷体
            ]
            
            for font_path in font_paths:
                if os.path.exists(font_path):
                    try:
                        return pygame.font.Font(font_path, size)
                    except:
                        continue
            
            # 如果系统字体都不可用，尝试使用freetype
            try:
                font = pygame.freetype.SysFont('Microsoft YaHei', size)
                return font
            except:
                try:
                    font = pygame.freetype.SysFont('SimHei', size)
                    return font
                except:
                    # 最后回退到默认字体
                    return pygame.font.Font(None, size)
        
        font = get_chinese_font(22)
        small_font = get_chinese_font(18)
        
        if self.organisms:
            avg_generation = sum(org.generation for org in self.organisms) / len(self.organisms)
            max_generation = max(org.generation for org in self.organisms)
            avg_energy = sum(org.energy for org in self.organisms) / len(self.organisms)
            avg_age = sum(org.age for org in self.organisms) / len(self.organisms)
        else:
            avg_generation = max_generation = avg_energy = avg_age = 0
        
        info_text = [
            f"时间: {self.time}",
            f"种群数量: {len(self.organisms)}",
            f"环境温度: {self.environment.temperature:.1f}°C",
            f"环境毒性: {self.environment.toxicity*100:.1f}%",
            f"最高世代: {max_generation}",
            f"平均世代: {avg_generation:.1f}",
            f"平均能量: {avg_energy:.1f}",
            f"平均年龄: {avg_age:.1f}"
        ]
        
        # 绘制信息面板背景
        info_bg = pygame.Surface((250, len(info_text) * 25 + 10))
        info_bg.set_alpha(180)
        info_bg.fill((0, 0, 0))
        surface.blit(info_bg, (5, 5))
        
        # 绘制信息文本
        for i, text in enumerate(info_text):
            color = (255, 255, 255) if i < 4 else (200, 200, 255)
            text_surface = font.render(text, True, color)
            surface.blit(text_surface, (10, 10 + i * 25))
    
    def draw_charts(self):
        """绘制进化图表（按C键生成）"""
        if len(self.population_history) < 10:
            print("数据不足，需要更多时间步骤才能生成有意义的图表")
            return
        
        # 设置matplotlib中文字体
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        plt.figure(figsize=(15, 10))
        plt.suptitle('生物进化模拟 - 数据分析', fontsize=16, fontweight='bold')
        
        # 1. 种群数量变化曲线
        plt.subplot(2, 3, 1)
        plt.plot(self.population_history, 'b-', linewidth=2)
        plt.title('种群数量变化', fontsize=12, fontweight='bold')
        plt.xlabel('时间步骤')
        plt.ylabel('生物数量')
        plt.grid(True, alpha=0.3)
        
        # 2. 特征进化趋势图
        plt.subplot(2, 3, 2)
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
        trait_names = ['体型', '速度', '感知', '繁殖', '防御', '寿命', '新陈代谢', '毒素抗性']
        
        for i, trait in enumerate(Trait):
            if trait in self.trait_history and len(self.trait_history[trait]) > 5:
                plt.plot(self.trait_history[trait], color=colors[i], 
                        label=trait_names[i], linewidth=1.5)
        
        plt.title('特征进化趋势', fontsize=12, fontweight='bold')
        plt.xlabel('时间步骤')
        plt.ylabel('特征值 (0-1)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        plt.grid(True, alpha=0.3)
        
        # 3. 生物特征雷达图
        if self.organisms:
            ax = plt.subplot(2, 3, 3, polar=True)
            traits = list(Trait)
            values = []
            for trait in traits:
                trait_avg = sum(org.dna[trait] for org in self.organisms) / len(self.organisms)
                values.append(trait_avg)
            
            # 闭合图形
            values += values[:1]
            angles = [n / float(len(traits)) * 2 * math.pi for n in range(len(traits))]
            angles += angles[:1]
            
            ax.plot(angles, values, 'o-', linewidth=2, color='blue')
            ax.fill(angles, values, alpha=0.25, color='blue')
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(trait_names, fontsize=8)
            ax.set_ylim(0, 1)
            plt.title('当前种群平均特征', fontsize=12, fontweight='bold', pad=20)
        
        # 4. 世代进化图
        if self.generation_history:
            plt.subplot(2, 3, 4)
            avg_gens = [g[0] for g in self.generation_history]
            max_gens = [g[1] for g in self.generation_history]
            
            plt.plot(avg_gens, 'g-', label='平均世代', linewidth=2)
            plt.plot(max_gens, 'r-', label='最高世代', linewidth=2)
            plt.title('世代进化', fontsize=12, fontweight='bold')
            plt.xlabel('时间步骤')
            plt.ylabel('世代数')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # 5. 死亡原因分析
        plt.subplot(2, 3, 5)
        causes = list(self.deaths_by_cause.keys())
        counts = list(self.deaths_by_cause.values())
        cause_names = ['能量耗尽', '寿命到期', '环境危害']
        
        if sum(counts) > 0:
            plt.pie(counts, labels=cause_names, autopct='%1.1f%%', startangle=90)
            plt.title('死亡原因分析', fontsize=12, fontweight='bold')
        
        # 6. 适应性分布
        if self.organisms:
            plt.subplot(2, 3, 6)
            energies = [org.energy for org in self.organisms]
            ages = [org.age for org in self.organisms]
            
            plt.scatter(ages, energies, alpha=0.6, c='green', s=30)
            plt.title('当前种群适应性分布', fontsize=12, fontweight='bold')
            plt.xlabel('年龄')
            plt.ylabel('能量')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('evolution_charts.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("进化图表已生成并保存为 'evolution_charts.png'")
        plt.close()

# 创建模拟
sim = EvolutionSimulation(150, 100)
scale_x = WIDTH / sim.width
scale_y = HEIGHT / sim.height

# 主循环
running = True
paused = False
fast_mode = False
show_charts = False

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                paused = not paused
            elif event.key == pygame.K_f:
                fast_mode = not fast_mode
            elif event.key == pygame.K_c:
                show_charts = True
            elif event.key == pygame.K_r:
                # 重置模拟
                sim = EvolutionSimulation(150, 100)
            # 环境参数控制
            elif event.key == pygame.K_q:
                # 增加温度
                sim.environment.temperature = min(50, sim.environment.temperature + 2)
            elif event.key == pygame.K_a:
                # 降低温度
                sim.environment.temperature = max(-10, sim.environment.temperature - 2)
            elif event.key == pygame.K_w:
                # 增加毒性
                sim.environment.toxicity = min(1.0, sim.environment.toxicity + 0.05)
            elif event.key == pygame.K_s:
                # 降低毒性
                sim.environment.toxicity = max(0, sim.environment.toxicity - 0.05)
            elif event.key == pygame.K_e:
                # 增加食物生成量
                sim.environment.food_multiplier = getattr(sim.environment, 'food_multiplier', 1.0)
                sim.environment.food_multiplier = min(3.0, sim.environment.food_multiplier + 0.2)
            elif event.key == pygame.K_d:
                # 减少食物生成量
                sim.environment.food_multiplier = getattr(sim.environment, 'food_multiplier', 1.0)
                sim.environment.food_multiplier = max(0.2, sim.environment.food_multiplier - 0.2)
    
    if not paused:
        if fast_mode:
            # 快速模式：一次更新多步
            for _ in range(10):
                sim.update()
        else:
            sim.update()
    
    # 绘制
    screen.fill((0, 0, 0))
    sim.draw(screen, scale_x, scale_y)
    
    # 显示控制提示
    # 使用相同的中文字体加载函数
    def get_chinese_font_for_controls(size):
        font_paths = [
            'C:/Windows/Fonts/msyh.ttc',  # 微软雅黑
            'C:/Windows/Fonts/simhei.ttf',  # 黑体
            'C:/Windows/Fonts/simsun.ttc',  # 宋体
        ]
        
        for font_path in font_paths:
            if os.path.exists(font_path):
                try:
                    return pygame.font.Font(font_path, size)
                except:
                    continue
        
        try:
            font = pygame.freetype.SysFont('Microsoft YaHei', size)
            return font
        except:
            return pygame.font.Font(None, size)
    
    font = get_chinese_font_for_controls(24)
    controls = [
        "空格键: 暂停/继续",  
        "F键: 快速模式",
        "C键: 生成图表",    
        "R键: 重置模拟",
        "Q/A键: ",
        "升高/降低温度",
        "W/S键: ",
        "增加/减少毒性",
        "E/D键: ",
        "增加/减少食物量"
    ]
    
    for i, text in enumerate(controls):
        text_surface = font.render(text, True, (200, 200, 200))
        screen.blit(text_surface, (WIDTH - 200, 10 + i * 25))
    
    pygame.display.flip()
    clock.tick(30)
    
    # 生成图表
    if show_charts:
        sim.draw_charts()
        show_charts = False

pygame.quit()