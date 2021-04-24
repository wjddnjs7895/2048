# 2048 game client file

import numpy as np
import pygame, sys, random
from pygame.locals import *

# game functions
def get_empty() : 
    empty_lst = []
    for x in range(WIDTH) : 
        for y in range(HEIGHT) : 
            if not BOARD[x,y] : empty_lst.append((x,y))
    return empty_lst

def random_generate(empty_lst) : 
    try :
        idx = random.randrange(0, len(empty_lst))
        BOARD[empty_lst[idx][0], empty_lst[idx][1]] = 2
        return True
    except : 
        return False

def move(dir) : 
    temp = BOARD.copy()
    if dir % 2 == 0 : # dir == 0, 2
        for x in range(HEIGHT) : 
            temp_lst = []
            idx = 0 if dir == 0 else -1
            for y in range(WIDTH) : 
                if BOARD[x,y] : temp_lst.append(BOARD[x,y])
                BOARD[x,y] = 0
            sum_lst = sum_line(temp_lst,dir)
            for value in sum_lst : 
                BOARD[x,idx] = value
                idx -= DX[dir]
    else : 
        for y in range(HEIGHT) : 
            temp_lst = []
            idx = 0 if dir == 1 else -1
            for x in range(WIDTH) : 
                if BOARD[x,y] : temp_lst.append(BOARD[x,y])
                BOARD[x,y] = 0
            sum_lst = sum_line(temp_lst,dir)
            for value in sum_lst : 
                BOARD[idx,y] = value
                idx += DY[dir]
    if not np.array_equal(temp, BOARD) : 
        if not random_generate(get_empty()) : 
            end_game()

def sum_line(lst, dir) :
    if dir >= 2 : lst.reverse()
        
    idx = 0
    result_lst = []
    if len(lst) == 1 : return lst
    while idx < len(lst): 
        try : 
            if lst[idx] == lst[idx + 1] : 
                result_lst.append(lst[idx] + lst[idx + 1])
                SCORE[0] = lst[idx] + lst[idx + 1]
                idx += 1
            else : 
                result_lst.append(lst[idx])
        except : 
            result_lst.append(lst[idx])
        idx += 1
    return result_lst

def end_game() : 
    pygame.quit()
    sys.exit()

# game main start function
def game_start() : 
    pygame.init()

    WINDOWWIDTH = 450
    WINDOWHEIGHT = 450
    windowSurface = pygame.display.set_mode((WINDOWWIDTH, WINDOWHEIGHT), 0, 32)
    pygame.display.set_caption('2048')

    COLOR_DIC = {'COLOR_0' : (213,204,197),
        'COLOR_2' : (236,227,218),
        'COLOR_4' : (235,224,204),
        'COLOR_8' : (232,180,130),
        'COLOR_16' : (232,154,108),
        'COLOR_32' : (230,131,102),
        'COLOR_64' : (228,103,71),
        'COLOR_128' : (233,208,126)}
    COLOR_BACKGROUND = (185,173,161)
    COLOR_WHITE = (249,246,241)
    COLOR_BLACK = (117,110,102)

    FONT_SIZE = 60
    font = pygame.font.SysFont(None,FONT_SIZE)

    random_generate(get_empty())

    while True : 
        for event in pygame.event.get() : 
            if event.type == QUIT : 
                pygame.quit()
                sys.exit()
            if event.type == KEYDOWN : 
                if event.key == K_LEFT : 
                    move(1)
                if event.key == K_UP : 
                    move(0)
                if event.key == K_RIGHT : 
                    move(3)
                if event.key == K_DOWN : 
                    move(2)

        windowSurface.fill(COLOR_BACKGROUND)

        for y in range(WIDTH) : 
            for x in range(HEIGHT) : 
                if BOARD[x][y] <= 64 : 
                    pygame.draw.rect(windowSurface, COLOR_DIC['COLOR_'+str(int(BOARD[x][y]))], [WINDOWWIDTH/45 * (11 * x + 1), WINDOWHEIGHT/45 * (11 * y+1),WINDOWWIDTH/45 * 10, WINDOWHEIGHT/45 * 10])
                else : 
                    pygame.draw.rect(windowSurface, COLOR_DIC['COLOR_128'], [WINDOWWIDTH/45 * (11 * x + 1), WINDOWHEIGHT/45 * (11 * y+1),WINDOWWIDTH/45 * 10, WINDOWHEIGHT/45 * 10])

                if BOARD[x][y] != 0 and BOARD[x][y] <= 4 : 
                    img = font.render(str(int(BOARD[x][y])),True,COLOR_BLACK)
                    windowSurface.blit(img, [WINDOWWIDTH/45 * (11 * x + 6)-FONT_SIZE*len(str(int(BOARD[x][y])))/5, WINDOWHEIGHT/45 * (11 * y+6)-FONT_SIZE/3])
                elif BOARD[x][y] >= 8: 
                    img = font.render(str(int(BOARD[x][y])),True,COLOR_WHITE)
                    windowSurface.blit(img, [WINDOWWIDTH/45 * (11 * x + 6)-FONT_SIZE*len(str(int(BOARD[x][y])))/5, WINDOWHEIGHT/45 * (11 * y+6)-FONT_SIZE/3])
        
        pygame.display.update()


# game settings and constant factors
DX = [-1, 0, 1, 0]
DY = [0, 1, 0, -1]
WIDTH = HEIGHT = 4
BOARD = np.zeros((WIDTH, HEIGHT))
SCORE = [2]

game_start()

'''
while True : 
    dir = int(input())
    random_generate(get_empty())
    move(dir)

    for x in range(WIDTH) : 
        for y in range(HEIGHT) : 
            print(BOARD[x,y],end = '  ')
        print("\n")
'''