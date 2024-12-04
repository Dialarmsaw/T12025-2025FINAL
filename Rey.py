def main():

    def needle():

        import pygame
        import math
        import random
        import matplotlib.pyplot as plt
        import numpy as np

        pygame.init()

        WIDTH, HEIGHT = 800, 600
        BG_COLOR = (255, 195, 125) 
        LINE_COLOR = (0, 0, 0)
        PICK_LENGTH = 40
        LINE_SPACING = 40 

        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("oopid picks")

        def draw_lines(line_spacing):
            for i in range(0, WIDTH, line_spacing):
                pygame.draw.line(screen, LINE_COLOR, (i, 0), (i, HEIGHT))

        def check_cross(x, y, angle, line_spacing):
            x_end = x + PICK_LENGTH * math.cos(angle)
            y_end = y + PICK_LENGTH * math.sin(angle)
            
            if (x // line_spacing) != (x_end // line_spacing):
                return True
            return False

        def create_plot(x_data, y_data):
            plt.ion() 

            fig, ax = plt.subplots()
            scatter = ax.scatter(x_data, y_data)

            fig.canvas.draw()

            background = fig.canvas.copy_from_bbox(ax.bbox)

            return fig, ax, scatter, background

        def update_plot(x_new, y_new, x_data, y_data, scatter, ax, background, fig):
            x_data.append(x_new)
            y_data.append(y_new)

            fig.canvas.restore_region(background)

            scatter.set_offsets(np.c_[x_data, y_data])

            ax.draw_artist(scatter)

            fig.canvas.blit(ax.bbox)
            fig.canvas.flush_events()

            ax.set_xlim(0, max(x_data)+10)
            ax.set_ylim(0, max(y_data)+0.5)

        def main():
            screen.fill(BG_COLOR)
            draw_lines(LINE_SPACING)

            x_data = [0]
            y_data = [0]
            fig, ax, scatter, background = create_plot(x_data, y_data)
            total_picks = 0
            crosses = 0
            probability = 0
            while True:
                x = random.uniform(0, WIDTH)
                y = random.uniform(0, HEIGHT)
                angle = random.uniform(0, 2 * math.pi)
                x_end = x + PICK_LENGTH * math.cos(angle)
                y_end = y + PICK_LENGTH * math.sin(angle)

                total_picks += 1
                if check_cross(x, y, angle, LINE_SPACING):
                    crosses += 1
                    pygame.draw.line(screen, (0, 255, 0), (x, y), (x_end, y_end))
                else:
                    pygame.draw.line(screen, (255, 0, 0), (x, y), (x_end, y_end))
                
                # Update the screen
                pygame.display.flip()

                # Calculate and print the probability
                if crosses > 0:
                    probability = crosses / total_picks
                    print(f"Total picks: {total_picks}, crosses: {crosses}, % touching: {probability:.4f}")
                
                update_plot(total_picks, probability, x_data, y_data, scatter, ax, background, fig)
                # Event handling
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return 
        main()
    
    def chess():
        import pygame
        import os
        import time
        pygame.init()
        global run
        WIDTH, HEIGHT = 800, 800
        ROWS, COLS = 8, 8
        SQ_SIZE = WIDTH // COLS
        WHITE = (255, 255, 255)
        BLACK = (0, 0, 0)
        GRAY = (128, 128, 128)
        SOFT_RED = (225, 135, 135)

        def load_images():
            pieces = ['wp', 'bp', 'wr', 'br', 'wn', 'bn', 'wb', 'bb', 'wq', 'bq', 'wk', 'bk']
            for piece in pieces:
                IMAGES[piece] = pygame.transform.scale(pygame.image.load(os.path.join('images', piece + '.png')), (SQ_SIZE, SQ_SIZE))

        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption('Chess')

        def main():
            run = True
            clock = pygame.time.Clock()
            board = create_board()
            selected_piece = None
            player_turn = 'w'
            load_images()

            while run:
                clock.tick(60)
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        run = False
                    elif event.type == pygame.MOUSEBUTTONDOWN:
                        pos = pygame.mouse.get_pos()
                        row, col = pos[1] // SQ_SIZE, pos[0] // SQ_SIZE
                        if selected_piece:
                            if is_valid_move(selected_piece, (row, col), board, player_turn):
                                run = move_piece(selected_piece, (row, col), board, run)
                                player_turn = 'b' if player_turn == 'w' else 'w' #turn switch
                            selected_piece = None
                        else:
                            if board[row][col] and board[row][col][0] == player_turn:
                                selected_piece = (row, col)
                                
                draw_board(screen, board)
                if selected_piece:
                    show_valid(selected_piece, board, player_turn, screen)
                pygame.display.flip()

            pygame.quit()

        def create_board():
            board = [[None for _ in range(COLS)] for _ in range(ROWS)]
            for col in range(COLS):
                board[1][col] = 'bp'
                board[6][col] = 'wp'
            pieces = ['r', 'n', 'b', 'q', 'k', 'b', 'n', 'r']
            for col in range(COLS):
                board[0][col] = 'b' + pieces[col]
                board[7][col] = 'w' + pieces[col]
            return board

        def draw_board(screen, board):
            colors = [WHITE, GRAY]
            for row in range(ROWS):
                for col in range(COLS):
                    color = colors[((row + col) % 2)]
                    pygame.draw.rect(screen, color, (col * SQ_SIZE, row * SQ_SIZE, SQ_SIZE, SQ_SIZE))
                    piece = board[row][col]
                    if piece:
                        screen.blit(IMAGES[piece], (col * SQ_SIZE, row * SQ_SIZE))

        def is_valid_move(piece, target, board, player_turn):
            src_row, src_col = piece
            dest_row, dest_col = target
            piece_type = board[src_row][src_col][1]

            if board[dest_row][dest_col] and board[dest_row][dest_col][0] == player_turn:
                return False

            #pawn move logic
            if piece_type == 'p':
                direction = -1 if player_turn == 'w' else 1
                if src_col == dest_col:  #moving forward
                    if dest_row == src_row + direction and board[dest_row][dest_col] is None:
                        return True
                    elif dest_row == src_row + 2 * direction and src_row in [1, 6] and board[src_row + direction][src_col] is None and board[dest_row][dest_col] is None:
                        return True
                elif abs(dest_col - src_col) == 1 and dest_row == src_row + direction and board[dest_row][dest_col] and board[dest_row][dest_col][0] != player_turn: #capture
                    return True

            #rook move logic
            elif piece_type == 'r':
                if src_row == dest_row or src_col == dest_col:
                    if not any(board[r][c] for r, c in squares_between(piece, target)):
                        return True

            #knight move logic
            elif piece_type == 'n':
                if (abs(src_row - dest_row), abs(src_col - dest_col)) in [(2, 1), (1, 2)]:
                    return True

            #bishop move logic
            elif piece_type == 'b':
                if abs(src_row - dest_row) == abs(src_col - dest_col):
                    if not any(board[r][c] for r, c in squares_between(piece, target)):
                        return True

            #queen move logic
            elif piece_type == 'q':
                if src_row == dest_row or src_col == dest_col or abs(src_row - dest_row) == abs(src_col - dest_col):
                    if not any(board[r][c] for r, c in squares_between(piece, target)):
                        return True

            #king move logic
            elif piece_type == 'k':
                if max(abs(src_row - dest_row), abs(src_col - dest_col)) == 1:
                    return True

            return False

        def squares_between(src, dest):
            src_row, src_col = src
            dest_row, dest_col = dest
            squares = []
            if src_row == dest_row:
                for c in range(min(src_col, dest_col) + 1, max(src_col, dest_col)):
                    squares.append((src_row, c))
            elif src_col == dest_col:
                for r in range(min(src_row, dest_row) + 1, max(src_row, dest_row)):
                    squares.append((r, src_col))
            elif abs(src_row - dest_row) == abs(src_col - dest_col):
                step = 1 if dest_row > src_row else -1
                for i in range(1, abs(dest_row - src_row)):
                    squares.append((src_row + i * step, src_col + i * (1 if dest_col > src_col else -1)))
            return squares

        def move_piece(piece, target, board, run):
            src_row, src_col = piece
            dest_row, dest_col = target
            if board[dest_row][dest_col] == "wk" or board[dest_row][dest_col] == "bk":
                run = win(board, dest_row, dest_col)
            board[dest_row][dest_col] = board[src_row][src_col]
            board[src_row][src_col] = None
            return run
            

        def show_valid(selected_piece, board, player_turn, screen):
            for rows in range(len(board)):
                for cols in range (len(board[0])):
                    if is_valid_move(selected_piece, (rows, cols), board, player_turn):
                        pygame.draw.rect(screen, SOFT_RED, (cols * SQ_SIZE, rows * SQ_SIZE, SQ_SIZE, SQ_SIZE))
                    piece = board[rows][cols]
                    if piece:
                        screen.blit(IMAGES[piece], (cols * SQ_SIZE, rows * SQ_SIZE))

        def is_checkmate():
            pass

        def win(board, king_row, king_col):
            run=False
            if board[king_row][king_col][0] == "w":
                print("black wins!")
            else:
                print("white wins!")
            return run


        if __name__ == "__main__":
            IMAGES = {}
            main()

    def conways():
        import pygame
        import numpy as np
        import time

        #cells are either 1 or 0, 0 is dead, 1 is alive.

        def gol(board):
            new_board = np.copy(board)
            rows, cols = board.shape
            neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
            for row in range(rows):
                for col in range(cols):
                    alive_neighbors = 0
                    for dr, dc in neighbors:
                        r, c = row + dr, col + dc
                        if 0 <= r < rows and 0 <= c < cols:
                            alive_neighbors += board[r, c]

                    if board[row, col] == 1:
                        if alive_neighbors < 2 or alive_neighbors > 3:
                            new_board[row, col] = 0
                    else:
                        if alive_neighbors == 3:
                            new_board[row, col] = 1
            return new_board

        def checkEdges(board):
            rows, cols = board.shape
            top_edge = board[0, :]
            bottom_edge = board[rows-1, :]
            left_edge = board[:, 0]
            right_edge = board[:, cols-1]

            if np.any(top_edge == 1) or np.any(bottom_edge == 1):
                board = np.vstack(([0]*cols, board, [0]*cols))
            if np.any(left_edge == 1) or np.any(right_edge == 1):
                board = np.hstack((np.zeros((board.shape[0], 1)), board, np.zeros((board.shape[0], 1))))
            
            return board

        def draw_board(screen, array, WIDTH, HEIGHT, DEAD_COLOR, ALIVE_COLOR, LINE_COLOR):
            screen.fill(DEAD_COLOR)
            
            rows, cols = array.shape
            CELL_SIZE = min(WIDTH // cols, HEIGHT // rows)
            
            for row in range(rows):
                for col in range(cols):
                    color = ALIVE_COLOR if array[row, col] == 1 else DEAD_COLOR
                    pygame.draw.rect(screen, color, (col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE))
            
            for row in range(rows + 1):
                pygame.draw.line(screen, LINE_COLOR, (0, row * CELL_SIZE), (WIDTH, row * CELL_SIZE))
            for col in range(cols + 1):
                pygame.draw.line(screen, LINE_COLOR, (col * CELL_SIZE, 0), (col * CELL_SIZE, HEIGHT))
            
            pygame.display.flip()

        def splash_screen(screen, WIDTH, HEIGHT):
            font = pygame.font.Font(None, 36)
            welcome_text = "Welcome to Conway's Game of Life!"
            instructions_text = "Click on cells to switch them from alive to dead or vice versa."
            instructions_text2 = "Press 'Space' to pause and play. Press any button to begin."
            
            welcome_surf = font.render(welcome_text, True, (255, 255, 255))
            instructions_surf = font.render(instructions_text, True, (255, 255, 255))
            instructions_surf2 = font.render(instructions_text2, True, (255, 255, 255))

            offset = -200
            screen.blit(welcome_surf, (WIDTH // 2 - welcome_surf.get_width() // 2, (HEIGHT // 2 - welcome_surf.get_height() // 2 - 40) + offset))
            screen.blit(instructions_surf, (WIDTH // 2 - instructions_surf.get_width() // 2, (HEIGHT // 2 - instructions_surf.get_height() // 2) + offset))
            screen.blit(instructions_surf2, (WIDTH // 2 - instructions_surf2.get_width() // 2, (HEIGHT // 2 - instructions_surf2.get_height() // 2 + 40) + offset))
            pygame.display.flip()

        def main(board):
            pygame.init()
            DEAD_COLOR = (60, 60, 65)
            ALIVE_COLOR = (210, 210, 120)
            LINE_COLOR = (0, 0, 0)
            WIDTH, HEIGHT = 800, 800
            screen = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption("Game of Life")
            draw_board(screen, board, WIDTH, HEIGHT, DEAD_COLOR, ALIVE_COLOR, LINE_COLOR)
            splash_screen(screen, WIDTH, HEIGHT)
            
            waiting = True
            while waiting:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        waiting = False
                        pygame.quit()
                        return
                    elif event.type == pygame.KEYDOWN or event.type == pygame.MOUSEBUTTONDOWN:
                        waiting = False
            
            running = True
            paused = True
            
            while running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_SPACE:
                            paused = not paused
                    elif event.type == pygame.MOUSEBUTTONDOWN:
                        x, y = event.pos
                        rows, cols = board.shape
                        CELL_SIZE = min(WIDTH // cols, HEIGHT // rows)
                        col, row = x // CELL_SIZE, y // CELL_SIZE
                        board[row, col] = 1 if board[row, col] == 0 else 0

                if not paused:
                    board = checkEdges(board)
                    draw_board(screen, board, WIDTH, HEIGHT, DEAD_COLOR, ALIVE_COLOR, LINE_COLOR)
                    board = gol(board)
                    time.sleep(0.1)
                else:
                    draw_board(screen, board, WIDTH, HEIGHT, DEAD_COLOR, ALIVE_COLOR, LINE_COLOR)
            
            pygame.quit()

        #starting board:
        board = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                        [0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],])

        main(board)

    s = input("Hello! Welcome to the code of the best coder in this class. There are 3 games you can play, a simulation of dropping needles called buffons needle (1), a really crummy chess game that I coded in an hour (2), or conways game of life (3). Input a number corrisponding with what you want, then hit RETURN >>> ")
    if s==1:
        needle()
    elif s==2:
        chess()
    else:
        conways()
