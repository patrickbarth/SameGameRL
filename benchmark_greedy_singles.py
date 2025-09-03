"""
Benchmarking script to profile GreedySinglesBot performance and identify bottlenecks.

This script analyzes the performance of the GreedySinglesBot across different board
sizes and game states to identify optimization opportunities.
"""

import cProfile
import pstats
import time
import statistics
from typing import Dict, List, Tuple
from io import StringIO

from samegamerl.agents.greedy_singles_bot import GreedySinglesBot
from samegamerl.agents.bot_utils import (
    find_valid_moves, 
    count_singles_after_move,
    simulate_move,
    count_singles
)
from samegamerl.game.game_config import GameFactory, GameConfig
from samegamerl.game.game import Game


def create_test_board(config: GameConfig, fill_percentage: float = 0.8) -> List[List[int]]:
    """Create a test board with specified fill percentage for consistent benchmarking."""
    game = Game(config)
    board = game.get_board()
    
    # Simulate some moves to create a more realistic mid-game state
    num_moves = int(config.num_rows * config.num_cols * (1 - fill_percentage) * 0.1)
    bot = GreedySinglesBot()
    
    for _ in range(num_moves):
        action = bot.select_action(board)
        if action is None:
            break
        row, col = action
        game.move((row, col))
        board = game.get_board()
    
    return board


def benchmark_function(func, *args, iterations: int = 100) -> Dict[str, float]:
    """Benchmark a function and return timing statistics."""
    times = []
    
    for _ in range(iterations):
        start_time = time.perf_counter()
        result = func(*args)
        end_time = time.perf_counter()
        times.append(end_time - start_time)
    
    return {
        'mean': statistics.mean(times),
        'median': statistics.median(times),
        'stdev': statistics.stdev(times) if len(times) > 1 else 0,
        'min': min(times),
        'max': max(times)
    }


def profile_bot_components(board: List[List[int]]) -> Dict[str, Dict[str, float]]:
    """Profile individual components of the bot's decision-making process."""
    print(f"Profiling board size: {len(board)}x{len(board[0])}")
    
    # Profile find_valid_moves
    valid_moves_stats = benchmark_function(find_valid_moves, board, iterations=1000)
    
    # Get valid moves for further testing
    valid_moves = find_valid_moves(board)
    if not valid_moves:
        return {'error': 'No valid moves available'}
    
    # Profile count_singles_after_move for a representative move
    test_move = valid_moves[0]
    singles_stats = benchmark_function(
        count_singles_after_move, board, test_move[0], test_move[1], iterations=100
    )
    
    # Profile simulate_move
    simulate_stats = benchmark_function(
        simulate_move, board, test_move[0], test_move[1], iterations=100
    )
    
    # Profile count_singles on current board
    count_stats = benchmark_function(count_singles, board, iterations=1000)
    
    # Profile full bot select_action
    bot = GreedySinglesBot()
    bot_stats = benchmark_function(bot.select_action, board, iterations=10)
    
    return {
        'find_valid_moves': valid_moves_stats,
        'count_singles_after_move': singles_stats,
        'simulate_move': simulate_stats,
        'count_singles': count_stats,
        'bot_select_action': bot_stats,
        'board_info': {
            'size': f"{len(board)}x{len(board[0])}",
            'valid_moves_count': len(valid_moves),
            'current_singles': count_singles(board)
        }
    }


def detailed_profile_bot_action(board: List[List[int]]) -> None:
    """Run detailed cProfile analysis on bot action selection."""
    bot = GreedySinglesBot()
    
    print(f"\n=== Detailed Profile for {len(board)}x{len(board[0])} board ===")
    
    # Create a StringIO object to capture profile output
    profile_stream = StringIO()
    profiler = cProfile.Profile()
    
    # Profile the bot's select_action method
    profiler.enable()
    for _ in range(10):  # Run multiple times for better statistics
        bot.select_action(board)
    profiler.disable()
    
    # Generate and display profile statistics
    stats = pstats.Stats(profiler, stream=profile_stream)
    stats.sort_stats('cumulative')
    stats.print_stats(20)  # Show top 20 functions
    
    print(profile_stream.getvalue())


def analyze_complexity_scaling():
    """Analyze how performance scales with board size."""
    print("\n=== Performance Scaling Analysis ===")
    
    configs = [
        GameFactory.small(),   # 5x5, 2 colors
        GameFactory.medium(),  # 8x8, 3 colors
        GameFactory.large(),   # 15x15, 5 colors
        GameFactory.custom(20, 20, 6)  # Extra large
    ]
    
    scaling_results = []
    
    for config in configs:
        board = create_test_board(config)
        results = profile_bot_components(board)
        
        if 'error' not in results:
            scaling_results.append({
                'config': config,
                'board_size': len(board) * len(board[0]),
                'results': results
            })
    
    # Print scaling analysis
    print("\nPerformance vs Board Size:")
    print("Size\t| ValidMoves | SinglesCheck | SimMove  | FullBot")
    print("-" * 60)
    
    for result in scaling_results:
        size = result['board_size']
        vm_time = result['results']['find_valid_moves']['mean'] * 1000
        sc_time = result['results']['count_singles_after_move']['mean'] * 1000
        sm_time = result['results']['simulate_move']['mean'] * 1000
        bot_time = result['results']['bot_select_action']['mean'] * 1000
        
        print(f"{size:4d}\t| {vm_time:8.3f}ms | {sc_time:9.3f}ms | {sm_time:7.3f}ms | {bot_time:6.3f}ms")
    
    return scaling_results


def identify_bottlenecks(scaling_results: List[Dict]) -> None:
    """Analyze results to identify the biggest bottlenecks."""
    print("\n=== Bottleneck Analysis ===")
    
    if not scaling_results:
        print("No scaling results available for analysis")
        return
    
    # Analyze the largest board size for bottleneck identification
    largest_result = max(scaling_results, key=lambda x: x['board_size'])
    results = largest_result['results']
    
    # Calculate relative time spent in each component
    bot_total_time = results['bot_select_action']['mean']
    
    component_times = {
        'find_valid_moves': results['find_valid_moves']['mean'],
        'count_singles_after_move': results['count_singles_after_move']['mean'],
        'simulate_move': results['simulate_move']['mean'],
        'count_singles': results['count_singles']['mean']
    }
    
    print(f"Analysis for {largest_result['config'].num_rows}x{largest_result['config'].num_cols} board:")
    print(f"Total bot action time: {bot_total_time * 1000:.3f}ms")
    print("\nComponent breakdown:")
    
    sorted_components = sorted(component_times.items(), key=lambda x: x[1], reverse=True)
    
    for component, time_taken in sorted_components:
        percentage = (time_taken / bot_total_time * 100) if bot_total_time > 0 else 0
        print(f"  {component:25s}: {time_taken * 1000:7.3f}ms ({percentage:5.1f}%)")
    
    # Estimate how many times each operation is called per bot action
    valid_moves_count = results['board_info']['valid_moves_count']
    print(f"\nEstimated operations per bot action:")
    print(f"  Valid moves found: {valid_moves_count}")
    print(f"  Singles calculations: {valid_moves_count} (once per move)")
    print(f"  Board simulations: {valid_moves_count} (once per move)")


def recommend_optimizations():
    """Provide optimization recommendations based on analysis."""
    print("\n=== Optimization Recommendations ===")
    
    recommendations = [
        "1. **Cache Board Analysis**: Implement memoization for repeated board states",
        "2. **Incremental Singles Counting**: Instead of full board scans, track singles changes",
        "3. **Early Termination**: Stop evaluation when a move with 0 singles is found",
        "4. **Move Ordering**: Evaluate moves in order of likely success (larger groups first)",
        "5. **Lazy Evaluation**: Only simulate moves that pass initial heuristics",
        "6. **Board Representation**: Consider more efficient data structures (numpy arrays)",
        "7. **Parallel Processing**: Evaluate multiple moves concurrently",
        "8. **Approximate Methods**: Use heuristics instead of exact simulation for some cases"
    ]
    
    for rec in recommendations:
        print(rec)


def main():
    """Run comprehensive benchmarking analysis."""
    print("=== GreedySinglesBot Performance Benchmark ===")
    
    # Run scaling analysis
    scaling_results = analyze_complexity_scaling()
    
    # Run detailed profiling on different board sizes
    configs_to_profile = [GameFactory.medium(), GameFactory.large()]
    
    for config in configs_to_profile:
        board = create_test_board(config)
        detailed_profile_bot_action(board)
    
    # Identify bottlenecks
    identify_bottlenecks(scaling_results)
    
    # Provide optimization recommendations
    recommend_optimizations()


if __name__ == "__main__":
    main()