# -*- coding: utf-8 -*-
# 导入所有需要的库
import pygame
import numpy as np
import random
import time
import collections
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Input
from tensorflow.keras.optimizers import Adam
import pickle # 用于保存和加载缓冲区
import os # 用于检查文件是否存在

# --- START: Game2048Env Class (游戏环境) ---
# 这个类包含 2048 游戏的所有规则和状态管理
class Game2048Env:
    def __init__(self, size=4):
        self.size = size
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.score = 0
        self.game_over = False
        self.actions = {0: "Up", 1: "Down", 2: "Left", 3: "Right"}

    def reset(self):
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.score = 0
        self.game_over = False
        self._add_random_tile()
        self._add_random_tile()
        return self.board.copy()

    def _add_random_tile(self):
        empty_cells = [(r, c) for r in range(self.size) for c in range(self.size) if self.board[r, c] == 0]
        if not empty_cells:
            return False
        r, c = random.choice(empty_cells)
        self.board[r, c] = np.random.choice([2, 4], p=[0.9, 0.1])
        return True

    def _move_line(self, line):
        new_line_list = [x for x in line if x != 0]
        new_line_list.extend([0] * (self.size - len(new_line_list)))

        score_added = 0
        temp_line = new_line_list[:]
        i = 0
        while i < len(temp_line) - 1:
            if temp_line[i] != 0 and temp_line[i] == temp_line[i+1]:
                temp_line[i] *= 2
                score_added += temp_line[i]
                temp_line.pop(i+1)
                temp_line.append(0)
            i += 1
        return np.array(temp_line, dtype=int), score_added

    def step(self, action):
        if self.game_over:
            return self.board.copy(), 0, True, {"message": "Game is over"}

        original_board = np.copy(self.board)
        score_added = 0

        if action == 0: # Up
            board_to_process = self.board
        elif action == 1: # Down
            board_to_process = np.rot90(self.board, k=2)
        elif action == 2: # Left
            board_to_process = np.rot90(self.board, k=-1)
        elif action == 3: # Right
            board_to_process = np.rot90(self.board, k=1)
        else:
            return self.board.copy(), 0, self.game_over, {"message": "Invalid action"}

        new_board_processed = np.zeros_like(board_to_process)
        for i in range(self.size):
            line = board_to_process[:, i]
            processed_line, added = self._move_line(line)
            new_board_processed[:, i] = processed_line
            score_added += added

        if action == 0: # Up
             self.board = new_board_processed
        elif action == 1: # Down
             self.board = np.rot90(new_board_processed, k=2)
        elif action == 2: # Left
             self.board = np.rot90(new_board_processed, k=1)
        elif action == 3: # Right
             self.board = np.rot90(new_board_processed, k=-1)

        if not np.array_equal(self.board, original_board):
            self.score += score_added
            self._add_random_tile()
            self.game_over = self._is_game_over()
        else:
             self.game_over = self._is_game_over()

        return self.board.copy(), score_added, self.game_over, {}

    def _is_game_over(self):
        if (self.board == 0).any():
            return False

        for r in range(self.size):
            for c in range(self.size - 1):
                if self.board[r, c] != 0 and self.board[r, c] == self.board[r, c+1]:
                    return False

        for r in range(self.size - 1):
            for c in range(self.size):
                if self.board[r, c] != 0 and self.board[r, c] == self.board[r+1, c]:
                    return False
        return True

    def render(self): # 控制台渲染，Pygame有自己的渲染
        print("-" * (self.size * 6 + 1))
        for row in self.board:
            print("|", end="")
            for cell in row:
                print(f"{cell: >5}", end="|")
            print()
            print("-" * (self.size * 6 + 1))

    def get_state(self):
        return self.board.copy()

    def get_score(self):
        return self.score

    def is_game_over(self):
        return self.game_over

    def get_available_actions(self):
        available_actions = []
        original_board = np.copy(self.board)

        for action in range(4):
            temp_env = Game2048Env()
            temp_env.board = np.copy(self.board)
            temp_env.size = self.size
            _, _, _, _ = temp_env.step(action) # 调用step，不关心返回值

            if not np.array_equal(temp_env.board, original_board):
                available_actions.append(action)

        return available_actions
# --- END: Game2048Env Class ---


# --- START: ReplayBuffer Class (经验回放缓冲区) ---
# 这个类用于存储和随机采样AI的经验
class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer = collections.deque(maxlen=buffer_size)
        self.buffer_size = buffer_size

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        actual_batch_size = min(batch_size, len(self.buffer))
        batch = random.sample(self.buffer, actual_batch_size)
        return batch

    def __len__(self):
        return len(self.buffer)
# --- END: ReplayBuffer Class ---


# --- START: Buffer Save/Load Functions (缓冲区文件存取) ---
# 这些函数用于将经验回放缓冲区的内容保存到文件和从文件加载
def save_buffer(buffer, filename="dqn_2048_replay_buffer.pkl"):
    """
    将经验回放缓冲区保存到文件
    :param buffer: ReplayBuffer 对象
    :param filename: 保存的文件名 (以 .pkl 结尾)
    """
    try:
        # 保存 deque 对象，而不是 ReplayBuffer 实例
        with open(filename, 'wb') as f:
            pickle.dump(buffer.buffer, f)
        print(f"Experience replay buffer saved to {filename}")
    except Exception as e:
        print(f"Error saving buffer to {filename}: {e}")

def load_buffer(filename="dqn_2048_replay_buffer.pkl", buffer_size=100000):
    """
    从文件加载经验回放缓冲区
    :param filename: 加载的文件名
    :param buffer_size: 缓冲区的最大容量 (加载后可能需要重新设置或初始化新的buffer)
    :return: 加载的 deque 对象，如果文件不存在或加载失败则返回 None
    """
    if not os.path.exists(filename):
        print(f"Buffer file not found: {filename}")
        return None
    try:
        with open(filename, 'rb') as f:
            loaded_deque = pickle.load(f)
        print(f"Experience replay buffer loaded from {filename}")
        # 可以检查加载的 deque 的 maxlen 是否与 buffer_size 匹配，或者直接使用它
        # loaded_deque = collections.deque(loaded_deque, maxlen=buffer_size) # 确保maxlen正确
        return loaded_deque
    except Exception as e:
        print(f"Error loading buffer from {filename}: {e}")
        # 如果加载失败，删除损坏的文件 (可选)
        # try: os.remove(filename) except: pass
        return None
# --- END: Buffer Save/Load Functions ---


# --- START: Neural Network Model Functions (AI模型) ---
# 这些函数用于构建 AI 的神经网络模型
def preprocess_state(board):
    """
    预处理游戏棋盘状态，转换为神经网络输入格式
    - 取 log2 转换非零值
    - 增加一个通道维度
    :param board: numpy 数组形式的 4x4 棋盘状态
    :return: 预处理后的状态，形状为 (4, 4, 1)
    """
    processed_board = np.zeros_like(board, dtype=np.float32)
    non_zero_mask = board != 0
    processed_board[non_zero_mask] = np.log2(board[non_zero_mask])
    return np.expand_dims(processed_board, axis=-1)

GRID_SIZE = 4 # 重新定义，确保在函数外也可用
NUM_ACTIONS = 4

def build_dqn_model(input_shape=(GRID_SIZE, GRID_SIZE, 1), num_actions=NUM_ACTIONS):
    """
    构建 DQN 的神经网络模型
    :param input_shape: 输入状态的形状
    :param num_actions: 可能的行动数量
    :return: 编译好的 Keras 模型
    """
    model = Sequential([
        Input(shape=input_shape),
        Conv2D(filters=32, kernel_size=2, activation='relu', padding='same'),
        Conv2D(filters=64, kernel_size=2, activation='relu', padding='same'),
        Flatten(),
        Dense(units=128, activation='relu'),
        Dense(units=num_actions, activation='linear')
    ])

    # 编译模型，定义优化器和学习率
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse') # loss='mse' 是占位符

    return model

# 定义模型权重文件名
MODEL_WEIGHTS_FILENAME = "dqn_2048_weights.h5"

def save_model_weights(model, filename=MODEL_WEIGHTS_FILENAME):
    """保存模型权重"""
    try:
        model.save_weights(filename)
        print(f"Model weights saved to {filename}")
    except Exception as e:
        print(f"Error saving model weights to {filename}: {e}")

def load_model_weights(model, filename=MODEL_WEIGHTS_FILENAME):
    """从文件加载模型权重"""
    if not os.path.exists(filename):
        print(f"Model weights file not found: {filename}")
        return False
    try:
        # 需要先构建模型结构，然后才能加载权重
        model.load_weights(filename)
        print(f"Model weights loaded from {filename}")
        return True
    except Exception as e:
        print(f"Error loading model weights from {filename}: {e}")
        # 如果加载失败，可能模型结构不匹配或文件损坏
        return False

# --- END: Neural Network Model Functions ---


# --- START: Pygame Visualization Code (可视化界面) ---
# 这个部分使用 Pygame 绘制游戏界面

# 颜色定义 (RGB)
COLOR_BACKGROUND = (250, 248, 239)
COLOR_GRID_BACKGROUND = (187, 173, 160)
COLOR_TEXT_LIGHT = (249, 246, 242) # 用于大数字 (>= 8)
COLOR_TEXT_DARK = (119, 110, 101) # 用于小数字 (< 8)

# 方块颜色 (根据数字定义不同的颜色)
TILE_COLORS = {
    0: (205, 193, 180), # Empty
    2: (238, 228, 218),
    4: (237, 224, 200),
    8: (242, 177, 121),
    16: (245, 149, 99),
    32: (246, 124, 95),
    64: (246, 94, 59),
    128: (237, 207, 114),
    256: (237, 204, 97),
    512: (237, 200, 80),
    1024: (237, 197, 63),
    2048: (237, 194, 46),
    4096: (60, 58, 50),
    8192: (60, 58, 50),
    16384: (60, 58, 50),
    32768: (60, 58, 50),
}

# 屏幕尺寸和布局
TILE_SIZE = 100
MARGIN = 15
WINDOW_WIDTH = GRID_SIZE * TILE_SIZE + (GRID_SIZE + 1) * MARGIN
WINDOW_HEIGHT = WINDOW_WIDTH + 100 # 额外空间用于显示得分

# 字体变量 (加载后会在这里赋值)
font_24 = None
font_8_16_32 = None
font_64_128 = None
font_256_512 = None
font_1024_2048 = None
font_score = None
font_game_over = None

def load_fonts():
    """加载所需的字体"""
    global font_24, font_8_16_32, font_64_128, font_256_512, font_1024_2048, font_score, font_game_over
    try:
        font_path = pygame.font.match_font('arial,dejavusans,sans') # 尝试查找常见字体
        if font_path:
             # 根据方块数值加载不同大小的字体
            font_24 = pygame.font.Font(font_path, 55)
            font_8_16_32 = pygame.font.Font(font_path, 50)
            font_64_128 = pygame.font.Font(font_path, 45)
            font_256_512 = pygame.font.Font(font_path, 40)
            font_1024_2048 = pygame.font.Font(font_path, 35)
            font_score = pygame.font.Font(font_path, 30)
            font_game_over = pygame.font.Font(font_path, 60)
        else:
            print("Warning: TrueType font not found, using default Pygame font.")
             # 如果找不到 TrueType 字体，使用默认字体
            font_24 = pygame.font.SysFont(None, 60)
            font_8_16_32 = pygame.font.SysFont(None, 55)
            font_64_128 = pygame.font.SysFont(None, 50)
            font_256_512 = pygame.font.SysFont(None, 45)
            font_1024_2048 = pygame.font.SysFont(None, 40)
            font_score = pygame.font.SysFont(None, 35)
            font_game_over = pygame.font.SysFont(None, 65)

    except Exception as e:
        print(f"Error loading font: {e}. Using default Pygame font.")
         # 出现错误时也使用默认字体
        font_24 = pygame.font.SysFont(None, 60)
        font_8_16_32 = pygame.font.SysFont(None, 55)
        font_64_128 = pygame.font.SysFont(None, 50)
        font_256_512 = pygame.font.SysFont(None, 45)
        font_1024_2048 = pygame.font.SysFont(None, 40)
        font_score = pygame.font.SysFont(None, 35)
        font_game_over = pygame.font.SysFont(None, 65)


def get_tile_color(value):
    return TILE_COLORS.get(value, TILE_COLORS[0])

def get_text_color(value):
    if value >= 8:
        return COLOR_TEXT_LIGHT
    return COLOR_TEXT_DARK

def get_tile_font(value):
    # 确保 load_fonts() 已经被调用
    if value <= 4:
        return font_24
    elif value <= 32:
        return font_8_16_32
    elif value <= 128:
        return font_64_128
    elif value <= 512:
        return font_256_512
    elif value <= 2048:
        return font_1024_2048
    else:
        return font_1024_2048


def draw_board(screen, board, score):
    screen.fill(COLOR_BACKGROUND)

    grid_rect = pygame.Rect(MARGIN, MARGIN + 100, WINDOW_WIDTH - 2 * MARGIN, WINDOW_WIDTH - 2 * MARGIN)
    pygame.draw.rect(screen, COLOR_GRID_BACKGROUND, grid_rect, border_radius=5)

    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            tile_value = board[r, c]
            tile_color = get_tile_color(tile_value)
            text_color = get_text_color(tile_value)

            x = MARGIN + c * (TILE_SIZE + MARGIN)
            y = MARGIN + 100 + r * (TILE_SIZE + MARGIN)

            pygame.draw.rect(screen, tile_color, (x, y, TILE_SIZE, TILE_SIZE), border_radius=5)

            if tile_value != 0:
                font_to_use = get_tile_font(tile_value)
                if font_to_use:
                    text_surface = font_to_use.render(str(tile_value), True, text_color)
                    text_rect = text_surface.get_rect(center=(x + TILE_SIZE / 2, y + TILE_SIZE / 2))
                    screen.blit(text_surface, text_rect)

    if font_score:
        score_surface = font_score.render(f"Score: {score}", True, COLOR_TEXT_DARK)
        score_rect = score_surface.get_rect(left=MARGIN, top=MARGIN)
        screen.blit(score_surface, score_rect)

def draw_game_over(screen):
    overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 128))
    screen.blit(overlay, (0, 0))

    if font_game_over:
        game_over_surface = font_game_over.render("Game Over!", True, COLOR_TEXT_LIGHT)
        game_over_rect = game_over_surface.get_rect(center=(WINDOW_WIDTH / 2, WINDOW_HEIGHT / 2))
        screen.blit(game_over_surface, game_over_rect)


# --- Pygame 主循环函数 (用于可视化) ---
# 这个函数用于运行带有 Pygame 界面的游戏，可以用来观察 AI 的表现
# 可以在训练后加载模型权重，让可视化显示训练好的 AI
def run_game_visualization(trained_model=None):
    print("Starting the Pygame visualization...")

    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("2048 AI Training Visualization")

    load_fonts() # 加载字体

    env = Game2048Env()
    state = env.reset()
    print("Game environment reset.")

    running = True
    clock = pygame.time.Clock()

    ai_action_delay = 0.1 # 控制AI行动速度 (秒)
    last_action_time = time.time()

    # AI 代理决策函数
    def get_ai_action(current_state, model):
        available_actions = env.get_available_actions()
        if not available_actions:
             return None

        if model is None:
            # 如果没有提供训练好的模型，使用随机行动
            action = random.choice(available_actions)
        else:
            # 使用训练好的模型进行预测
            processed_state = preprocess_state(current_state)
            input_state = np.expand_dims(processed_state, axis=0)
            q_values = model.predict(input_state, verbose=0)
            # 选择 Q 值最高的有效行动
            # 这是一个简单的策略，更高级的可能会考虑有效性
            # 为了简单，这里先选择 Q 值最高的，即使是无效行动 step() 会处理
            action = np.argmax(q_values[0])

            # 如果需要确保行动有效，可以在这里过滤
            # valid_q_values = [(q_values[0][i], i) for i in available_actions]
            # if valid_q_values:
            #     action = max(valid_q_values)[1]
            # else:
            #     action = random.choice([0,1,2,3]) # 没有有效行动就随机选

        return action


    # --- 游戏循环 ---
    while running:
        # --- 事件处理 ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            # 这里可以添加键盘事件，让你手动控制游戏
            # if event.type == pygame.KEYDOWN:
            #     action = None
            #     if event.key == pygame.K_UP: action = 0
            #     elif event.key == pygame.K_DOWN: action = 1
            #     elif event.key == pygame.K_LEFT: action = 2
            #     elif event.key == pygame.K_RIGHT: action = 3
            #
            #     if action is not None and not env.is_game_over():
            #          state, reward, done, _ = env.step(action)


        # --- AI 自动行动逻辑 ---
        if not env.is_game_over():
            current_time = time.time()
            if current_time - last_action_time > ai_action_delay:
                # 将训练好的模型传递给 get_ai_action
                action = get_ai_action(state, trained_model)

                if action is not None:
                    state, reward, done, _ = env.step(action)
                    last_action_time = current_time
        # else: # 如果游戏结束，但窗口还开着，可以继续处理退出事件

        # --- 绘制和更新 ---
        draw_board(screen, state, env.get_score())

        if env.is_game_over():
             draw_game_over(screen)

        pygame.display.flip()

        clock.tick(60) # 控制帧率


    print("Exiting game visualization function.")
    pygame.quit()
    print("Pygame quit.")
# --- END: Pygame Visualization Code ---


# --- START: DQN Training Function (AI训练) ---
# 这个函数执行强化学习的训练过程
# 训练循环参数 (超参数)
BUFFER_SIZE = 100000 # 经验回放缓冲区的大小
BATCH_SIZE = 32      # 从缓冲区采样多少经验进行训练
GAMMA = 0.99         # 折扣因子
EPSILON_START = 1.0  # 探索率的起始值
EPSILON_END = 0.01   # 探索率的最终值
# epsilon 衰减步数可以根据训练总步骤数调整
EPSILON_DECAY_STEPS = 100000 # epsilon 从 start 衰减到 end 需要多少个游戏步骤 (调大衰减慢)
LEARNING_RATE = 0.001 # 神经网络的学习率
TRAINING_START_STEP = 1000 # 在缓冲区有多少经验后才开始训练
TARGET_UPDATE_FREQ = 1000 # 每隔多少个训练步骤更新一次目标网络
NUM_EPISODES = 5000 # 训练多少个游戏回合 (可以调整，越多效果可能越好)
MAX_STEPS_PER_EPISODE = 1000 # 每个游戏回合最多进行多少步 (防止无限循环)

# 定义文件保存路径
BUFFER_FILENAME = "dqn_2048_replay_buffer.pkl"
MODEL_WEIGHTS_FILENAME_TRAIN = "dqn_2048_weights_train.h5" # 训练时用的权重文件名

def train_dqn_agent():
    print("Starting DQN agent training...")

    # 1. 初始化环境、缓冲区和模型
    env = Game2048Env()
    buffer = ReplayBuffer(BUFFER_SIZE)

    # 尝试加载缓冲区
    loaded_deque = load_buffer(BUFFER_FILENAME, BUFFER_SIZE)
    if loaded_deque is not None:
        buffer.buffer = loaded_deque
        print(f"Initialized buffer with {len(buffer)} loaded experiences.")

    # 构建主网络和目标网络
    primary_network = build_dqn_model()
    target_network = build_dqn_model()

    # 尝试加载模型权重
    # 注意：load_model_weights 内部会处理文件是否存在和加载错误
    model_loaded = load_model_weights(primary_network, MODEL_WEIGHTS_FILENAME_TRAIN)

    # 如果加载了模型，目标网络权重也从主网络复制
    if model_loaded:
        target_network.set_weights(primary_network.get_weights())
        # TODO: 如果加载了，epsilon 和 total_steps 可能需要从一个保存的状态恢复
        # 这需要额外的逻辑来保存和加载训练状态 (epsilon, total_steps, episode_rewards等)
        print("Resuming training from loaded model and buffer (epsilon and steps may not be exact).")
    else:
         # 如果没有加载模型，初始时目标网络与主网络权重相同
         target_network.set_weights(primary_network.get_weights())
         print("Starting training from scratch.")


    optimizer = primary_network.optimizer

    # 如果没有从头开始，epsilon 和 total_steps 需要加载
    # 为了简化，这里先不实现保存/加载训练状态，每次运行训练都是从初始 epsilon/total_steps 开始
    epsilon = EPSILON_START
    total_steps = 0 # 总的训练步骤数 (用于 epsilon 衰减和目标网络更新)
    episode_rewards = [] # 训练过程中每个回合的总奖励记录

    # 2. 训练主循环 (按回合进行)
    for episode in range(NUM_EPISODES):
        state = env.reset()
        episode_reward = 0
        done = False
        step_count = 0 # 当前回合的步数

        # 3. 回合内的步骤循环
        while not done and step_count < MAX_STEPS_PER_EPISODE:
            total_steps += 1
            step_count += 1

            # 4. 行动选择 (epsilon-greedy 策略)
            # Epsilon 在每个训练步骤后衰减
            current_epsilon = max(EPSILON_END, EPSILON_START - (EPSILON_START - EPSILON_END) * total_steps / EPSILON_DECAY_STEPS)

            if random.random() < current_epsilon:
                available_actions = env.get_available_actions()
                if available_actions:
                     action = random.choice(available_actions)
                else:
                     action = random.randint(0, NUM_ACTIONS - 1) # 无有效行动时随机选一个无效的
            else:
                processed_state = preprocess_state(state)
                input_state = np.expand_dims(processed_state, axis=0)
                # 使用主网络进行预测
                q_values = primary_network.predict(input_state, verbose=0)
                # 选择 Q 值最高的行动
                action = np.argmax(q_values[0])


            # 5. 在环境中执行行动
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward

            # 6. 将经验存储到经验回放缓冲区
            buffer.add((state, action, reward, next_state, int(done)))

            # 7. 更新当前状态
            state = next_state

            # 8. 训练步骤 (从缓冲区采样并更新网络)
            if len(buffer) > TRAINING_START_STEP:
                batch = buffer.sample(BATCH_SIZE)
                # 确保采样批次大小等于 BATCH_SIZE，否则 tf.stack 会出错
                if len(batch) == BATCH_SIZE:
                    states_b, actions_b, rewards_b, next_states_b, dones_b = zip(*batch)

                    states_b = np.array(states_b)
                    actions_b = np.array(actions_b)
                    rewards_b = np.array(rewards_b)
                    next_states_b = np.array(next_states_b)
                    dones_b = np.array(dones_b)

                    processed_states_batch = np.array([preprocess_state(s) for s in states_b])
                    processed_next_states_batch = np.array([preprocess_state(ns) for ns in next_states_b])

                    with tf.GradientTape() as tape:
                        predicted_q_values = primary_network(processed_states_batch)
                        next_q_values = target_network(processed_next_states_batch)
                        max_next_q_values = tf.reduce_max(next_q_values, axis=1)
                        target_q_values = rewards_b + GAMMA * max_next_q_values * (1 - dones_b)

                        indices = tf.stack([tf.range(BATCH_SIZE), tf.cast(actions_b, tf.int32)], axis=1)
                        gathered_q_values = tf.gather_nd(predicted_q_values, indices)

                        loss = tf.keras.losses.MSE(target_q_values, gathered_q_values)


                    # 9. 反向传播和更新主网络权重
                    gradients = tape.gradient(loss, primary_network.trainable_variables)
                    optimizer.apply_gradients(zip(gradients, primary_network.trainable_variables))

                    # 10. 更新目标网络权重 (定期更新)
                    if total_steps % TARGET_UPDATE_FREQ == 0:
                        print(f"--- Updating target network at total step {total_steps} ---")
                        target_network.set_weights(primary_network.get_weights())

        # 回合结束后的日志记录
        episode_rewards.append(episode_reward)
        print(f"Episode {episode+1}/{NUM_EPISODES}: Total Reward = {episode_reward}, Steps = {step_count}, Epsilon = {current_epsilon:.4f}, Buffer Size = {len(buffer)}")

        # --- 保存缓冲区和模型权重 (定期) ---
        # 例如，每隔 100 个回合保存一次
        if (episode + 1) % 100 == 0:
             print(f"--- Saving buffer and model at end of episode {episode+1} ---")
             save_buffer(buffer, BUFFER_FILENAME)
             save_model_weights(primary_network, MODEL_WEIGHTS_FILENAME_TRAIN)


    print("Training finished.")
    # --- 训练结束时保存缓冲区和模型权重 ---
    print("--- Saving buffer and model at end of training ---")
    save_buffer(buffer, BUFFER_FILENAME)
    save_model_weights(primary_network, MODEL_WEIGHTS_FILENAME_TRAIN)


    return episode_rewards
# --- END: DQN Training Function ---


# --- START: Main Execution Block (程序入口) ---
# 这个块决定了运行脚本时会执行什么
if __name__ == "__main__":
    print("2048 AI Training and Visualization Script")
    print("Choose an option:")
    print("1: Run Pygame Visualization (Random or Trained AI)")
    print("2: Run DQN Training")
    print("Any other key: Exit")

    choice = input("Enter choice (1 or 2): ")

    if choice == '1':
        # 运行 Pygame 可视化
        print("\nStarting Pygame visualization...")
        # 在可视化时，可以选择加载训练好的模型
        print(f"Attempting to load model weights from {MODEL_WEIGHTS_FILENAME_TRAIN} for visualization...")
        # 构建一个模型实例用于加载权重
        visualization_model = build_dqn_model()
        model_loaded_for_viz = load_model_weights(visualization_model, MODEL_WEIGHTS_FILENAME_TRAIN)

        if model_loaded_for_viz:
             print("Using loaded model for visualization.")
             run_game_visualization(trained_model=visualization_model)
        else:
             print("Model weights not found. Using random AI for visualization.")
             run_game_visualization(trained_model=None) # 没有加载模型，传递 None

        print("\nPygame visualization finished.")

    elif choice == '2':
        # 运行 DQN 训练
        # 注意：训练过程不会显示 Pygame 窗口，可能会运行一段时间
        print("\nStarting DQN training...")
        # 清除可能残留的 TensorFlow 状态，防止重复创建网络等问题 (可选)
        tf.keras.backend.clear_session()
        # 调用训练函数
        train_rewards = train_dqn_agent()
        print("\nTraining Rewards per Episode:")
        print(train_rewards)
        print("\nTraining finished.")
    else:
        print("\nExiting script.")

# --- END: Main Execution Block ---