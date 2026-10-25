import javax.swing.*;
import java.awt.*;
import java.util.*;
import java.util.List;

public class MouseMazeApp extends JFrame {

    // ===================== ÉNUMÉRATIONS =====================
    public enum CellType { EMPTY, WALL, CHEESE, WATER, TRAP }

    public enum Action {
        UP(0, -1), DOWN(0, 1), LEFT(-1, 0), RIGHT(1, 0);
        public final int dx, dy;
        Action(int dx, int dy) { this.dx = dx; this.dy = dy; }
    }

    // ===================== LABYRINTHE =====================
    public static class Maze {
        private final int width;
        private final int height;
        private final CellType[][] grid;
        private final Point startPos = new Point(1, 1);
        private final Point cheesePos;

        public Maze(int width, int height) {
            this.width = (width % 2 == 0) ? width + 1 : width;
            this.height = (height % 2 == 0) ? height + 1 : height;
            this.grid = new CellType[this.height][this.width];
            this.cheesePos = new Point(this.width - 2, this.height - 2);

            generateBraidMazeDFS();
            populateObjects();
        }

        private void generateBraidMazeDFS() {
            // 1. Tout en murs
            for (int y = 0; y < height; y++) {
                Arrays.fill(grid[y], CellType.WALL);
            }
            // 2. DFS (Recursive Backtracker)
            Stack<Point> stack = new Stack<>();
            grid[1][1] = CellType.EMPTY;
            stack.push(new Point(1, 1));
            Random rng = new Random();

            while (!stack.isEmpty()) {
                Point current = stack.peek();
                List<Point> neighbors = new ArrayList<>();
                for (Action act : Action.values()) {
                    int nx = current.x + act.dx * 2;
                    int ny = current.y + act.dy * 2;
                    if (nx > 0 && nx < width - 1 && ny > 0 && ny < height - 1
                            && grid[ny][nx] == CellType.WALL) {
                        neighbors.add(new Point(nx, ny));
                    }
                }
                if (!neighbors.isEmpty()) {
                    Point chosen = neighbors.get(rng.nextInt(neighbors.size()));
                    grid[(current.y + chosen.y) / 2][(current.x + chosen.x) / 2] = CellType.EMPTY;
                    grid[chosen.y][chosen.x] = CellType.EMPTY;
                    stack.push(chosen);
                } else {
                    stack.pop();
                }
            }

            // 3. Percement de passages supplémentaires (~18 %)
            for (int y = 2; y < height - 2; y++) {
                for (int x = 2; x < width - 2; x++) {
                    if (grid[y][x] == CellType.WALL) {
                        boolean horizontalPass = (grid[y][x - 1] == CellType.EMPTY
                                && grid[y][x + 1] == CellType.EMPTY);
                        boolean verticalPass = (grid[y - 1][x] == CellType.EMPTY
                                && grid[y + 1][x] == CellType.EMPTY);
                        if ((horizontalPass || verticalPass) && rng.nextDouble() < 0.18) {
                            grid[y][x] = CellType.EMPTY;
                        }
                    }
                }
            }
        }

        private void populateObjects() {
            grid[cheesePos.y][cheesePos.x] = CellType.CHEESE;
            Random rng = new Random();

            List<Point> freeCells = new ArrayList<>();
            for (int y = 1; y < height - 1; y++) {
                for (int x = 1; x < width - 1; x++) {
                    if (grid[y][x] == CellType.EMPTY) {
                        if (Math.abs(x - startPos.x) + Math.abs(y - startPos.y) > 2
                                && !(x == cheesePos.x && y == cheesePos.y)) {
                            freeCells.add(new Point(x, y));
                        }
                    }
                }
            }
            Collections.shuffle(freeCells, rng);

            int waterCount = Math.max(3, freeCells.size() / 15);
            int trapCount  = Math.max(2, freeCells.size() / 20);
            int idx = 0;
            for (int i = 0; i < waterCount && idx < freeCells.size(); i++) {
                Point p = freeCells.get(idx++);
                grid[p.y][p.x] = CellType.WATER;
            }
            for (int i = 0; i < trapCount && idx < freeCells.size(); i++) {
                Point p = freeCells.get(idx++);
                grid[p.y][p.x] = CellType.TRAP;
            }
        }

        public boolean isWall(int x, int y) {
            if (x < 0 || x >= width || y < 0 || y >= height) return true;
            return grid[y][x] == CellType.WALL;
        }
        public CellType getCell(int x, int y) { return grid[y][x]; }
        public int getWidth() { return width; }
        public int getHeight() { return height; }
        public Point getStartPos() { return new Point(startPos); }
    }

    // ===================== ENVIRONNEMENT =====================
    public record StepResult(Point nextState, double reward, boolean isDone) {}

    public static class Environment {
        private final Maze maze;
        private final Set<Point> visitedWaterInEpisode = new HashSet<>();
        private final double rewardCheese = 300.0;
        private final double rewardWater  = 20.0;
        private final double penaltyTrap  = -40.0;
        private final double penaltyWall  = -3.0;
        private final double stepPenalty  = -0.1;

        public Environment(Maze maze) { this.maze = maze; }
        public void reset() { visitedWaterInEpisode.clear(); }

        public StepResult step(Point currentPos, Action action) {
            int tx = currentPos.x + action.dx;
            int ty = currentPos.y + action.dy;

            if (maze.isWall(tx, ty)) {
                return new StepResult(currentPos, penaltyWall, false);
            }
            Point nextPos = new Point(tx, ty);
            CellType cell = maze.getCell(tx, ty);

            if (cell == CellType.CHEESE) {
                return new StepResult(nextPos, rewardCheese, true);
            }
            if (cell == CellType.TRAP) {
                return new StepResult(nextPos, penaltyTrap, false);
            }
            if (cell == CellType.WATER && !visitedWaterInEpisode.contains(nextPos)) {
                visitedWaterInEpisode.add(nextPos);
                return new StepResult(nextPos, rewardWater, false);
            }
            return new StepResult(nextPos, stepPenalty, false);
        }
    }

    // ===================== AGENT (Q-Learning) =====================
    public static class MouseAgent {
        private final double[][][] qTable;
        private final double alpha = 0.3;
        private final double gamma = 0.95;
        private double epsilon = 1.0;
        private final Random rng = new Random();

        public MouseAgent(int width, int height) {
            this.qTable = new double[height][width][Action.values().length];
        }

        public Action chooseAction(Point s, boolean greedyOnly) {
            if (!greedyOnly && rng.nextDouble() < epsilon) {
                return Action.values()[rng.nextInt(Action.values().length)];
            }
            double maxQ = -Double.MAX_VALUE;
            List<Action> bestActions = new ArrayList<>();
            for (int i = 0; i < Action.values().length; i++) {
                double val = qTable[s.y][s.x][i];
                if (val > maxQ) {
                    maxQ = val;
                    bestActions.clear();
                    bestActions.add(Action.values()[i]);
                } else if (Math.abs(val - maxQ) < 1e-5) {
                    bestActions.add(Action.values()[i]);
                }
            }
            return bestActions.get(rng.nextInt(bestActions.size()));
        }

        public void updateQ(Point s, Action a, double reward, Point nextS, boolean isDone) {
            int aIdx = a.ordinal();
            double oldQ = qTable[s.y][s.x][aIdx];
            double maxNextQ = 0.0;
            if (!isDone) {
                maxNextQ = -Double.MAX_VALUE;
                for (int i = 0; i < Action.values().length; i++) {
                    if (qTable[nextS.y][nextS.x][i] > maxNextQ) {
                        maxNextQ = qTable[nextS.y][nextS.x][i];
                    }
                }
            }
            double target = reward + (isDone ? 0.0 : gamma * maxNextQ);
            qTable[s.y][s.x][aIdx] = oldQ + alpha * (target - oldQ);
        }

        public void decayEpsilon() {
            this.epsilon = Math.max(0.05, this.epsilon * 0.994);
        }
    }

    // ===================== FENÊTRE PRINCIPALE =====================
    private Maze maze;
    private Environment env;
    private MouseAgent agent;
    private Point mousePos;
    private int totalEpisodes = 0;
    private final JPanel canvas;
    private final JLabel lblStatus;
    private final JButton btnTrain;
    private final JButton btnRun;
    private final JButton btnRegen;

    public MouseMazeApp() {
        super("Labo n°2 : Souris dans le labyrinthe (Q-Learning)");
        initNewMaze(15, 15);
        mousePos = maze.getStartPos();

        canvas = new JPanel() {
            @Override
            protected void paintComponent(Graphics g) {
                super.paintComponent(g);
                renderGrid(g);
            }
        };
        canvas.setPreferredSize(new Dimension(540, 540));
        canvas.setBackground(new Color(30, 33, 40));

        JPanel south = new JPanel(new BorderLayout());
        JPanel buttons = new JPanel(new FlowLayout());

        btnTrain = new JButton("Entraîner (+50époques)");
        btnRun   = new JButton("Montrer le chemin");
        btnRegen = new JButton("Nouveau labyrinthe");
        lblStatus = new JLabel("Prêt. Époques : 0", SwingConstants.CENTER);
        lblStatus.setBorder(BorderFactory.createEmptyBorder(5, 5, 5, 5));

        btnTrain.addActionListener(e -> train(50));
        btnRun.addActionListener(e -> runVisualPath());
        btnRegen.addActionListener(e -> {
            initNewMaze(15, 15);
            mousePos = maze.getStartPos();
            totalEpisodes = 0;
            canvas.repaint();
            lblStatus.setText("Nouveau labyrinthe généré. Époques : 0");
        });

        buttons.add(btnTrain);
        buttons.add(btnRun);
        buttons.add(btnRegen);

        south.add(buttons, BorderLayout.NORTH);
        south.add(lblStatus, BorderLayout.SOUTH);

        add(canvas, BorderLayout.CENTER);
        add(south, BorderLayout.SOUTH);

        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        pack();
        setLocationRelativeTo(null);
    }

    private void initNewMaze(int w, int h) {
        this.maze  = new Maze(w, h);
        this.env   = new Environment(maze);
        this.agent = new MouseAgent(maze.getWidth(), maze.getHeight());
        this.mousePos = maze.getStartPos();
    }

    private void renderGrid(Graphics g) {
        int cellSize = Math.min(canvas.getWidth()  / maze.getWidth(),
                canvas.getHeight() / maze.getHeight());

        for (int y = 0; y < maze.getHeight(); y++) {
            for (int x = 0; x < maze.getWidth(); x++) {
                CellType cell = maze.getCell(x, y);
                switch (cell) {
                    case WALL    -> g.setColor(new Color(50, 50, 60));
                    case EMPTY   -> g.setColor(new Color(230, 230, 230));
                    case CHEESE  -> g.setColor(new Color(255, 215, 0));
                    case WATER   -> g.setColor(new Color(80, 160, 255));
                    case TRAP    -> g.setColor(new Color(200, 60, 60));
                }
                g.fillRect(x * cellSize, y * cellSize, cellSize, cellSize);
                g.setColor(new Color(180, 180, 180));
                g.drawRect(x * cellSize, y * cellSize, cellSize, cellSize);
            }
        }

        if (mousePos != null) {
            g.setColor(new Color(120, 80, 40));
            g.fillOval(mousePos.x * cellSize + cellSize / 4,
                    mousePos.y * cellSize + cellSize / 4,
                    cellSize / 2, cellSize / 2);
        }
    }

    private void setControlsEnabled(boolean enabled) {
        btnTrain.setEnabled(enabled);
        btnRun.setEnabled(enabled);
        btnRegen.setEnabled(enabled);
    }

    private void train(int episodes) {
        setControlsEnabled(false);
        new Thread(() -> {
            int cheeseReachedCount = 0;
            for (int i = 0; i < episodes; i++) {
                env.reset();
                Point s = maze.getStartPos();
                int steps = 0;
                boolean done = false;

                while (!done && steps < 2500) {
                    Action a = agent.chooseAction(s, false);
                    StepResult res = env.step(s, a);
                    agent.updateQ(s, a, res.reward(), res.nextState(), res.isDone());
                    s = res.nextState();
                    done = res.isDone();
                    steps++;
                    if (done && maze.getCell(s.x, s.y) == CellType.CHEESE) {
                        cheeseReachedCount++;
                    }
                    if (cheeseReachedCount > 0) {
                        agent.decayEpsilon();
                    }
                }
                totalEpisodes++;
            }
            final int wins = cheeseReachedCount;
            SwingUtilities.invokeLater(() -> {
                lblStatus.setText(String.format(
                        "Époques : %d | Fromage trouvé : %d fois | ε : %.2f",
                        totalEpisodes, wins, agent.epsilon));
                setControlsEnabled(true);
            });
        }).start();
    }

    private void runVisualPath() {
        setControlsEnabled(false);
        new Thread(() -> {
            mousePos = maze.getStartPos();
            SwingUtilities.invokeLater(canvas::repaint);

            Point prevPos = null;
            int loopCounter = 0;

            for (int step = 0; step < 120; step++) {
                try { Thread.sleep(40); } catch (InterruptedException ignored) {}

                Action bestAction = agent.chooseAction(mousePos, true);
                StepResult res = env.step(mousePos, bestAction);

                if (prevPos != null && prevPos.equals(res.nextState())) {
                    loopCounter++;
                } else {
                    loopCounter = 0;
                }
                prevPos = mousePos;
                mousePos = res.nextState();
                SwingUtilities.invokeLater(canvas::repaint);

                if (res.isDone() || loopCounter > 5) break;
            }
            SwingUtilities.invokeLater(() -> setControlsEnabled(true));
        }).start();
    }

    public static void main(String[] args) {
        SwingUtilities.invokeLater(() -> new MouseMazeApp().setVisible(true));
    }
}