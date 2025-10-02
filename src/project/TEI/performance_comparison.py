# src/project/performance_comparison.py

import time
import logging
import subprocess
import psutil
import os
from pathlib import Path
from typing import Dict, Any, List
import json

from project.pydantic_models import ProcessingConfig
from project.complete_pipeline_gpu import CompleteOptimizedPipeline
from project.tei_pipeline import TEIMilvusPipeline

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """Monitor system performance metrics during pipeline execution."""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.peak_memory = 0
        self.peak_cpu = 0
        self.monitoring = False
    
    def start_monitoring(self):
        """Start performance monitoring."""
        self.start_time = time.time()
        self.peak_memory = 0
        self.peak_cpu = 0
        self.monitoring = True
        logger.info("Performance monitoring started")
    
    def update_metrics(self):
        """Update performance metrics."""
        if not self.monitoring:
            return
        
        try:
            # Memory usage in MB
            memory_mb = psutil.virtual_memory().used / (1024 * 1024)
            self.peak_memory = max(self.peak_memory, memory_mb)
            
            # CPU percentage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            self.peak_cpu = max(self.peak_cpu, cpu_percent)
            
        except Exception as e:
            logger.warning(f"Error updating metrics: {e}")
    
    def stop_monitoring(self) -> Dict[str, Any]:
        """Stop monitoring and return results."""
        self.end_time = time.time()
        self.monitoring = False
        
        total_time = self.end_time - self.start_time if self.start_time else 0
        
        results = {
            "total_time": total_time,
            "peak_memory_mb": self.peak_memory,
            "peak_cpu_percent": self.peak_cpu,
        }
        
        logger.info(f"Performance monitoring stopped: {results}")
        return results


class PipelineComparison:
    """Compare performance between different RAG pipelines."""
    
    def __init__(self, input_dir: str, output_dir: str):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.results = {}
        
        # Create output directories
        Path(output_dir).mkdir(exist_ok=True)
        Path(output_dir + "/local_pipeline").mkdir(exist_ok=True)
        Path(output_dir + "/tei_pipeline").mkdir(exist_ok=True)
    
    def check_tei_availability(self) -> bool:
        """Check if TEI server is running."""
        try:
            import requests
            response = requests.get("http://localhost:8080/health", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"TEI server not available: {e}")
            return False
    
    def run_local_pipeline_test(self, config: ProcessingConfig) -> Dict[str, Any]:
        """Run the local embedding pipeline."""
        logger.info("=== TESTING LOCAL EMBEDDING PIPELINE ===")
        
        monitor = PerformanceMonitor()
        monitor.start_monitoring()
        
        try:
            pipeline = CompleteOptimizedPipeline(config, max_workers=2)
            
            # Run pipeline with performance monitoring
            start_time = time.time()
            pipeline.process_complete_pipeline(
                input_dir=self.input_dir,
                output_dir=self.output_dir + "/local_pipeline",
                use_bulk_import=False  # Skip bulk import for fair comparison
            )
            end_time = time.time()
            
            # Get final metrics
            perf_metrics = monitor.stop_monitoring()
            
            # Count processed chunks
            output_file = Path(self.output_dir + "/local_pipeline/prepared_data.json")
            chunk_count = 0
            if output_file.exists():
                with open(output_file, 'r') as f:
                    data = json.load(f)
                    chunk_count = len(data.get("rows", []))
            
            results = {
                "pipeline_type": "local_embedding",
                "total_time": end_time - start_time,
                "chunks_processed": chunk_count,
                "chunks_per_second": chunk_count / (end_time - start_time) if (end_time - start_time) > 0 else 0,
                "peak_memory_mb": perf_metrics["peak_memory_mb"],
                "peak_cpu_percent": perf_metrics["peak_cpu_percent"],
                "embedding_model": "sentence-transformers/all-mpnet-base-v2",
                "device": "mps" if hasattr(psutil, 'mps') else "cpu",
                "success": True
            }
            
            logger.info(f"Local pipeline results: {results}")
            return results
            
        except Exception as e:
            logger.error(f"Local pipeline failed: {e}")
            perf_metrics = monitor.stop_monitoring()
            return {
                "pipeline_type": "local_embedding",
                "success": False,
                "error": str(e),
                "peak_memory_mb": perf_metrics["peak_memory_mb"],
                "peak_cpu_percent": perf_metrics["peak_cpu_percent"]
            }
    
    def run_tei_pipeline_test(self, config: ProcessingConfig) -> Dict[str, Any]:
        """Run the TEI embedding pipeline."""
        logger.info("=== TESTING TEI EMBEDDING PIPELINE ===")
        
        # Check if TEI is available
        if not self.check_tei_availability():
            return {
                "pipeline_type": "tei_embedding",
                "success": False,
                "error": "TEI server not available at http://localhost:8080"
            }
        
        monitor = PerformanceMonitor()
        monitor.start_monitoring()
        
        try:
            pipeline = TEIMilvusPipeline(config, tei_endpoint="http://localhost:8080")
            
            # Run pipeline with performance monitoring
            start_time = time.time()
            pipeline.process_complete_pipeline(
                input_dir=self.input_dir,
                output_dir=self.output_dir + "/tei_pipeline",
                use_direct_insert=True
            )
            end_time = time.time()
            
            # Get final metrics
            perf_metrics = monitor.stop_monitoring()
            
            # Get chunk count from Milvus
            stats = pipeline.get_stats()
            chunk_count = stats.get("total_chunks", 0)
            
            results = {
                "pipeline_type": "tei_embedding",
                "total_time": end_time - start_time,
                "chunks_processed": chunk_count,
                "chunks_per_second": chunk_count / (end_time - start_time) if (end_time - start_time) > 0 else 0,
                "peak_memory_mb": perf_metrics["peak_memory_mb"],
                "peak_cpu_percent": perf_metrics["peak_cpu_percent"],
                "embedding_model": "huggingface-tei",
                "device": "docker_container",
                "success": True
            }
            
            logger.info(f"TEI pipeline results: {results}")
            return results
            
        except Exception as e:
            logger.error(f"TEI pipeline failed: {e}")
            perf_metrics = monitor.stop_monitoring()
            return {
                "pipeline_type": "tei_embedding",
                "success": False,
                "error": str(e),
                "peak_memory_mb": perf_metrics["peak_memory_mb"],
                "peak_cpu_percent": perf_metrics["peak_cpu_percent"]
            }
    
    def run_comparison(self, chunk_size: int = 1024, chunk_overlap: int = 256) -> Dict[str, Any]:
        """Run full comparison between pipelines."""
        logger.info("=== STARTING PIPELINE PERFORMANCE COMPARISON ===")
        
        config = ProcessingConfig(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        # Run local pipeline test
        local_results = self.run_local_pipeline_test(config)
        
        # Run TEI pipeline test
        tei_results = self.run_tei_pipeline_test(config)
        
        # Compile comparison results
        comparison = {
            "comparison_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "input_directory": self.input_dir,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "local_pipeline": local_results,
            "tei_pipeline": tei_results
        }
        
        # Add performance comparison if both succeeded
        if local_results.get("success") and tei_results.get("success"):
            comparison["performance_comparison"] = self._calculate_performance_diff(
                local_results, tei_results
            )
        
        # Save results
        results_file = Path(self.output_dir) / "performance_comparison.json"
        with open(results_file, 'w') as f:
            json.dump(comparison, f, indent=2)
        
        logger.info(f"Comparison results saved to: {results_file}")
        self._print_comparison_summary(comparison)
        
        return comparison
    
    def _calculate_performance_diff(self, local: Dict, tei: Dict) -> Dict[str, Any]:
        """Calculate performance differences between pipelines."""
        time_diff = ((tei["total_time"] - local["total_time"]) / local["total_time"]) * 100
        speed_diff = ((tei["chunks_per_second"] - local["chunks_per_second"]) / local["chunks_per_second"]) * 100
        memory_diff = ((tei["peak_memory_mb"] - local["peak_memory_mb"]) / local["peak_memory_mb"]) * 100
        
        return {
            "time_difference_percent": round(time_diff, 2),
            "speed_difference_percent": round(speed_diff, 2),
            "memory_difference_percent": round(memory_diff, 2),
            "faster_pipeline": "TEI" if tei["total_time"] < local["total_time"] else "Local",
            "more_memory_efficient": "TEI" if tei["peak_memory_mb"] < local["peak_memory_mb"] else "Local"
        }
    
    def _print_comparison_summary(self, comparison: Dict):
        """Print a summary of the comparison results."""
        print("\n" + "="*50)
        print("PERFORMANCE COMPARISON SUMMARY")
        print("="*50)
        
        local = comparison["local_pipeline"]
        tei = comparison["tei_pipeline"]
        
        if local.get("success") and tei.get("success"):
            print(f"Local Pipeline:")
            print(f"  Time: {local['total_time']:.2f}s")
            print(f"  Speed: {local['chunks_per_second']:.1f} chunks/sec")
            print(f"  Peak Memory: {local['peak_memory_mb']:.1f} MB")
            print(f"  Chunks: {local['chunks_processed']}")
            
            print(f"\nTEI Pipeline:")
            print(f"  Time: {tei['total_time']:.2f}s")
            print(f"  Speed: {tei['chunks_per_second']:.1f} chunks/sec")
            print(f"  Peak Memory: {tei['peak_memory_mb']:.1f} MB")
            print(f"  Chunks: {tei['chunks_processed']}")
            
            if "performance_comparison" in comparison:
                perf = comparison["performance_comparison"]
                print(f"\nComparison:")
                print(f"  Faster Pipeline: {perf['faster_pipeline']}")
                print(f"  Time Difference: {perf['time_difference_percent']:.1f}%")
                print(f"  Speed Difference: {perf['speed_difference_percent']:.1f}%")
                print(f"  Memory Difference: {perf['memory_difference_percent']:.1f}%")
        else:
            if not local.get("success"):
                print(f"Local Pipeline FAILED: {local.get('error', 'Unknown error')}")
            if not tei.get("success"):
                print(f"TEI Pipeline FAILED: {tei.get('error', 'Unknown error')}")
        
        print("="*50)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare RAG pipeline performance")
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="./comparison_results")
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--chunk-overlap", type=int, default=256)
    
    args = parser.parse_args()
    
    comparison = PipelineComparison(args.input_dir, args.output_dir)
    results = comparison.run_comparison(args.chunk_size, args.chunk_overlap)


if __name__ == "__main__":
    main()
    