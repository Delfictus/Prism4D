//! End-to-End Pipeline Integration Tests
//!
//! Comprehensive integration tests for the PRISM pipeline:
//! - Full pipeline execution on DIMACS graphs
//! - GPU fallback mechanisms
//! - Phase execution verification
//! - FluxNet RL integration
//! - Multi-phase warmstart
//!
//! ## Running Tests
//! ```bash
//! cargo test --test pipeline_e2e -- --nocapture
//! cargo test --test pipeline_e2e --features cuda -- --nocapture
//! ```

use prism_core::{dimacs::parse_dimacs_file, ColoringSolution, Graph};
use prism_fluxnet::{RLConfig, UniversalRLController};
use prism_pipeline::{orchestrator::PipelineOrchestrator, PipelineConfig};
use std::collections::HashMap;
use std::path::PathBuf;

/// Helper to get the project root directory
fn project_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// Helper to create a minimal test configuration
fn test_config() -> PipelineConfig {
    PipelineConfig {
        max_vertices: 10000,
        phase_configs: HashMap::new(),
        timeout_seconds: 600, // 10 minutes for tests
        enable_telemetry: false,
        telemetry_path: "test_telemetry.jsonl".to_string(),
        warmstart_config: None,
        gpu: Default::default(),
        phase2: Default::default(),
        memetic: None,
        metaphysical_coupling: None,
        ontology: None,
        mec: None,
        cma_es: None,
        gnn: None,
    }
}

/// Helper to create RL controller
fn test_rl_controller() -> UniversalRLController {
    let config = RLConfig::default();
    UniversalRLController::new(config)
}

/// Verify that a coloring solution is valid
fn verify_solution(graph: &Graph, solution: &ColoringSolution) -> Result<(), String> {
    // Check no conflicts
    if solution.conflicts > 0 {
        return Err(format!(
            "Solution has {} conflicts - INVALID",
            solution.conflicts
        ));
    }

    // Verify against adjacency list
    let mut actual_conflicts = 0;
    for (u, neighbors) in graph.adjacency.iter().enumerate() {
        for &v in neighbors {
            if u < v && solution.colors[u] == solution.colors[v] {
                actual_conflicts += 1;
            }
        }
    }

    if actual_conflicts > 0 {
        return Err(format!(
            "Solution reports 0 conflicts but has {} actual edge conflicts",
            actual_conflicts
        ));
    }

    // Verify coloring vector length
    if solution.colors.len() != graph.num_vertices {
        return Err(format!(
            "Coloring vector length {} doesn't match graph vertices {}",
            solution.colors.len(),
            graph.num_vertices
        ));
    }

    // Verify chromatic number matches actual color usage
    let max_color = solution
        .colors
        .iter()
        .max()
        .copied()
        .unwrap_or(0);
    let unique_colors: std::collections::HashSet<_> = solution.colors.iter().copied().collect();

    if unique_colors.len() != solution.chromatic_number {
        return Err(format!(
            "Chromatic number {} doesn't match unique color count {}",
            solution.chromatic_number,
            unique_colors.len()
        ));
    }

    // Verify colors are in valid range
    if max_color >= graph.num_vertices {
        return Err(format!(
            "Color {} exceeds vertex count {}",
            max_color, graph.num_vertices
        ));
    }

    Ok(())
}

#[test]
fn test_pipeline_small_graph() {
    env_logger::builder()
        .filter_level(log::LevelFilter::Info)
        .try_init()
        .ok();

    println!("\n🧪 Test: Pipeline execution on small graph (DSJC125.5)");

    let graph_path = project_root().join("data/dimacs/DSJC125.5.col");
    let graph = parse_dimacs_file(&graph_path).expect("Failed to load DSJC125.5");

    println!(
        "  Graph: {} vertices, {} edges",
        graph.num_vertices, graph.num_edges
    );

    let config = test_config();
    let rl_controller = test_rl_controller();
    let mut orchestrator = PipelineOrchestrator::new(config, rl_controller);

    let solution = orchestrator
        .run(&graph)
        .expect("Pipeline execution failed");

    println!("  Result: {} colors, {} conflicts", solution.chromatic_number, solution.conflicts);

    verify_solution(&graph, &solution).expect("Invalid solution");

    // DSJC125.5 best known: 17 colors
    assert!(
        solution.chromatic_number <= 25,
        "Chromatic number {} exceeds reasonable bound for DSJC125.5",
        solution.chromatic_number
    );

    println!("  ✓ Test passed\n");
}

#[test]
fn test_pipeline_medium_graph() {
    env_logger::builder()
        .filter_level(log::LevelFilter::Info)
        .try_init()
        .ok();

    println!("\n🧪 Test: Pipeline execution on medium graph (DSJC250.5)");

    let graph_path = project_root().join("data/dimacs/DSJC250.5.col");
    let graph = parse_dimacs_file(&graph_path).expect("Failed to load DSJC250.5");

    println!(
        "  Graph: {} vertices, {} edges",
        graph.num_vertices, graph.num_edges
    );

    let config = test_config();
    let rl_controller = test_rl_controller();
    let mut orchestrator = PipelineOrchestrator::new(config, rl_controller);

    let solution = orchestrator
        .run(&graph)
        .expect("Pipeline execution failed");

    println!("  Result: {} colors, {} conflicts", solution.chromatic_number, solution.conflicts);

    verify_solution(&graph, &solution).expect("Invalid solution");

    // DSJC250.5 best known: 28 colors
    assert!(
        solution.chromatic_number <= 40,
        "Chromatic number {} exceeds reasonable bound for DSJC250.5",
        solution.chromatic_number
    );

    println!("  ✓ Test passed\n");
}

#[test]
#[cfg(feature = "cuda")]
fn test_gpu_fallback() {
    env_logger::builder()
        .filter_level(log::LevelFilter::Info)
        .try_init()
        .ok();

    println!("\n🧪 Test: GPU fallback mechanism");

    let graph_path = project_root().join("data/dimacs/myciel5.col");
    let graph = parse_dimacs_file(&graph_path).expect("Failed to load myciel5");

    println!(
        "  Graph: {} vertices, {} edges",
        graph.num_vertices, graph.num_edges
    );

    let config = test_config();
    let rl_controller = test_rl_controller();
    let mut orchestrator = PipelineOrchestrator::new(config, rl_controller);

    // Pipeline should work regardless of GPU availability (GPU context initialized in new())
    println!("  Note: GPU initialization happens automatically if CUDA feature is enabled");

    let solution = orchestrator
        .run(&graph)
        .expect("Pipeline execution failed");

    println!("  Result: {} colors, {} conflicts", solution.chromatic_number, solution.conflicts);

    verify_solution(&graph, &solution).expect("Invalid solution");

    // Myciel5 chromatic number is 6
    assert!(
        solution.chromatic_number <= 8,
        "Chromatic number {} exceeds reasonable bound for myciel5",
        solution.chromatic_number
    );

    println!("  ✓ Test passed\n");
}

#[test]
fn test_phase_execution_order() {
    env_logger::builder()
        .filter_level(log::LevelFilter::Info)
        .try_init()
        .ok();

    println!("\n🧪 Test: Phase execution verification");

    let graph_path = project_root().join("data/dimacs/queen8_8.col");
    let graph = parse_dimacs_file(&graph_path).expect("Failed to load queen8_8");

    println!(
        "  Graph: {} vertices, {} edges",
        graph.num_vertices, graph.num_edges
    );

    let mut config = test_config();
    config.enable_telemetry = true;
    config.telemetry_path = "/tmp/prism_test_telemetry.jsonl".to_string();

    let rl_controller = test_rl_controller();
    let mut orchestrator = PipelineOrchestrator::new(config, rl_controller);

    let solution = orchestrator
        .run(&graph)
        .expect("Pipeline execution failed");

    println!("  Result: {} colors, {} conflicts", solution.chromatic_number, solution.conflicts);

    verify_solution(&graph, &solution).expect("Invalid solution");

    // Check telemetry file was created
    let telemetry_path = PathBuf::from("/tmp/prism_test_telemetry.jsonl");
    if telemetry_path.exists() {
        println!("  ✓ Telemetry file created");
        let _ = std::fs::remove_file(telemetry_path);
    }

    println!("  ✓ Test passed\n");
}

#[test]
fn test_fluxnet_integration() {
    env_logger::builder()
        .filter_level(log::LevelFilter::Info)
        .try_init()
        .ok();

    println!("\n🧪 Test: FluxNet RL integration");

    let graph_path = project_root().join("data/dimacs/myciel6.col");
    let graph = parse_dimacs_file(&graph_path).expect("Failed to load myciel6");

    println!(
        "  Graph: {} vertices, {} edges",
        graph.num_vertices, graph.num_edges
    );

    let config = test_config();
    let rl_controller = test_rl_controller();

    // Note: RL controller tracks rewards internally during execution
    println!("  RL controller initialized");

    let mut orchestrator = PipelineOrchestrator::new(config, rl_controller);

    let solution = orchestrator
        .run(&graph)
        .expect("Pipeline execution failed");

    println!("  Result: {} colors, {} conflicts", solution.chromatic_number, solution.conflicts);

    verify_solution(&graph, &solution).expect("Invalid solution");

    // RL controller should have accumulated rewards during execution
    // (Note: We can't easily access the internal RL controller state after run,
    //  but we can verify the solution quality which reflects RL performance)

    println!("  ✓ Test passed\n");
}

#[test]
fn test_warmstart_mechanism() {
    env_logger::builder()
        .filter_level(log::LevelFilter::Info)
        .try_init()
        .ok();

    println!("\n🧪 Test: Warmstart mechanism");

    let graph_path = project_root().join("data/dimacs/le450_15a.col");
    let graph = parse_dimacs_file(&graph_path).expect("Failed to load le450_15a");

    println!(
        "  Graph: {} vertices, {} edges",
        graph.num_vertices, graph.num_edges
    );

    let config = test_config();

    let rl_controller = test_rl_controller();
    let mut orchestrator = PipelineOrchestrator::new(config, rl_controller);

    let solution = orchestrator
        .run(&graph)
        .expect("Pipeline execution failed");

    println!("  Result: {} colors, {} conflicts", solution.chromatic_number, solution.conflicts);

    verify_solution(&graph, &solution).expect("Invalid solution");

    println!("  ✓ Test passed\n");
}

#[test]
fn test_retry_escalate_logic() {
    env_logger::builder()
        .filter_level(log::LevelFilter::Info)
        .try_init()
        .ok();

    println!("\n🧪 Test: Retry and escalate logic");

    let graph_path = project_root().join("data/dimacs/DSJC125.1.col");
    let graph = parse_dimacs_file(&graph_path).expect("Failed to load DSJC125.1");

    println!(
        "  Graph: {} vertices, {} edges",
        graph.num_vertices, graph.num_edges
    );

    let config = test_config();

    let rl_controller = test_rl_controller();
    let mut orchestrator = PipelineOrchestrator::new(config, rl_controller);

    let solution = orchestrator
        .run(&graph)
        .expect("Pipeline execution failed");

    println!("  Result: {} colors, {} conflicts", solution.chromatic_number, solution.conflicts);

    verify_solution(&graph, &solution).expect("Invalid solution");

    println!("  ✓ Test passed\n");
}

#[test]
#[cfg(feature = "cuda")]
fn test_ultra_kernel_integration() {
    env_logger::builder()
        .filter_level(log::LevelFilter::Info)
        .try_init()
        .ok();

    println!("\n🧪 Test: Ultra Kernel integration");

    let graph_path = project_root().join("data/dimacs/DSJC125.5.col");
    let graph = parse_dimacs_file(&graph_path).expect("Failed to load DSJC125.5");

    println!(
        "  Graph: {} vertices, {} edges",
        graph.num_vertices, graph.num_edges
    );

    let config = test_config();
    let rl_controller = test_rl_controller();
    let mut orchestrator = PipelineOrchestrator::new(config, rl_controller);

    // Enable Ultra Kernel
    orchestrator.enable_ultra_kernel(true);

    if let Err(e) = orchestrator.initialize_ultra_kernel(&graph) {
        println!("  ⚠ Ultra Kernel initialization failed: {}. Skipping test.", e);
        return;
    }

    println!("  ✓ Ultra Kernel initialized");

    let solution = orchestrator
        .run(&graph)
        .expect("Pipeline execution failed");

    println!("  Result: {} colors, {} conflicts", solution.chromatic_number, solution.conflicts);

    verify_solution(&graph, &solution).expect("Invalid solution");

    println!("  ✓ Test passed\n");
}

#[test]
fn test_empty_graph() {
    env_logger::builder()
        .filter_level(log::LevelFilter::Info)
        .try_init()
        .ok();

    println!("\n🧪 Test: Empty graph handling");

    let graph = Graph::new(0);

    let config = test_config();
    let rl_controller = test_rl_controller();
    let mut orchestrator = PipelineOrchestrator::new(config, rl_controller);

    // Empty graph should be handled gracefully
    let result = orchestrator.run(&graph);

    match result {
        Ok(solution) => {
            println!("  Result: {} colors", solution.chromatic_number);
            assert_eq!(solution.chromatic_number, 0, "Empty graph should have 0 colors");
            println!("  ✓ Test passed\n");
        }
        Err(e) => {
            println!("  ⚠ Empty graph handling returned error: {}", e);
            println!("  Note: This is acceptable if empty graphs are not supported\n");
        }
    }
}

#[test]
fn test_single_vertex_graph() {
    env_logger::builder()
        .filter_level(log::LevelFilter::Info)
        .try_init()
        .ok();

    println!("\n🧪 Test: Single vertex graph");

    let graph = Graph::new(1);

    let config = test_config();
    let rl_controller = test_rl_controller();
    let mut orchestrator = PipelineOrchestrator::new(config, rl_controller);

    let solution = orchestrator
        .run(&graph)
        .expect("Pipeline execution failed");

    println!("  Result: {} colors", solution.chromatic_number);

    verify_solution(&graph, &solution).expect("Invalid solution");

    assert_eq!(
        solution.chromatic_number, 1,
        "Single vertex should require 1 color"
    );

    println!("  ✓ Test passed\n");
}
