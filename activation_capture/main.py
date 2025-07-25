#!/usr/bin/env python3
"""
Main activation capture script

This script orchestrates the process of:
1. Retrieving questions from ChromaDB
2. Passing them to the inference endpoint
3. Capturing model activations
4. Storing the results

Usage:
    python -m activation_capture.main [options]
"""

import argparse
import logging
import sys
from typing import Optional, List, Dict, Any
from pathlib import Path
import time

from .config import config, ActivationCaptureConfig
from .question_retriever import QuestionRetriever
from .inference_client import InferenceClient
from .activation_storage import ActivationStorage

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ActivationCaptureRunner:
    """Main runner class for activation capture pipeline"""
    
    def __init__(self, custom_config: Optional[ActivationCaptureConfig] = None):
        """Initialize the activation capture runner
        
        Args:
            custom_config: Optional custom configuration
        """
        self.config = custom_config or config
        self.question_retriever = QuestionRetriever()
        self.inference_client = InferenceClient()
        self.activation_storage = ActivationStorage()
        
        logger.info("Initialized ActivationCaptureRunner")
    
    def run(self, 
            filter_type: str = "all",
            max_questions: Optional[int] = None,
            target_layers: Optional[List[int]] = None) -> Dict[str, Any]:
        """Run the complete activation capture pipeline
        
        Args:
            filter_type: Type of questions to process ("all", "censored", "uncensored", or category name)
            max_questions: Maximum number of questions to process
            target_layers: List of layer indices to capture activations from
            
        Returns:
            Dictionary containing run statistics
        """
        start_time = time.time()
        
        # Update configuration if provided
        if target_layers:
            self.config.target_layers = target_layers
        if max_questions:
            self.config.max_questions = max_questions
        
        logger.info(f"Starting activation capture run with filter_type='{filter_type}', "
                   f"max_questions={max_questions}, target_layers={target_layers}")
        
        try:
            # Step 1: Test connections
            logger.info("Testing connections...")
            self._test_connections()
            
            # Step 2: Retrieve questions
            logger.info("Retrieving questions from ChromaDB...")
            questions = self._retrieve_questions(filter_type, max_questions)
            
            if not questions:
                logger.warning("No questions retrieved. Exiting.")
                return {"status": "no_questions", "processed": 0, "errors": 0}
            
            logger.info(f"Retrieved {len(questions)} questions for processing")
            
            # Step 3: Process questions and capture activations
            logger.info("Starting activation capture process...")
            results = self._process_questions(questions)
            
            # Step 4: Generate summary statistics
            end_time = time.time()
            duration = end_time - start_time
            
            summary = {
                "status": "completed",
                "total_questions": len(questions),
                "processed": results["processed"],
                "errors": results["errors"],
                "duration_seconds": duration,
                "questions_per_second": results["processed"] / duration if duration > 0 else 0,
                "config": {
                    "target_layers": self.config.target_layers,
                    "save_format": self.config.save_format,
                    "compression": self.config.compression,
                    "model_name": self.config.model_name
                },
                "storage_stats": self.activation_storage.get_storage_stats()
            }
            
            logger.info(f"Activation capture completed. "
                       f"Processed {results['processed']}/{len(questions)} questions "
                       f"in {duration:.2f} seconds")
            
            return summary
            
        except Exception as e:
            logger.error(f"Error during activation capture run: {e}")
            raise
    
    def _test_connections(self):
        """Test connections to required services"""
        # Test ChromaDB connection
        try:
            self.question_retriever.connect()
            logger.info("✓ ChromaDB connection successful")
        except Exception as e:
            logger.error(f"✗ ChromaDB connection failed: {e}")
            raise
        
        # Test inference server connection
        if not self.inference_client.test_connection():
            raise ConnectionError("Failed to connect to inference server")
        logger.info("✓ Inference server connection successful")
    
    def _retrieve_questions(self, filter_type: str, max_questions: Optional[int]) -> List[Dict[str, Any]]:
        """Retrieve questions based on filter criteria"""
        if filter_type == "all":
            questions = self.question_retriever.get_all_questions(limit=max_questions)
        elif filter_type == "censored":
            questions = self.question_retriever.get_censored_questions(limit=max_questions)
        elif filter_type == "uncensored":
            questions = self.question_retriever.get_uncensored_questions(limit=max_questions)
        else:
            # Assume it's a category name
            questions = self.question_retriever.get_questions_by_category(
                filter_type, limit=max_questions
            )
        
        return questions
    
    def _process_questions(self, questions: List[Dict[str, Any]]) -> Dict[str, int]:
        """Process questions and capture activations
        
        Args:
            questions: List of question dictionaries
            
        Returns:
            Dictionary with processing statistics
        """
        processed = 0
        errors = 0
        
        for i, question_data in enumerate(questions):
            question_id = question_data['id']
            question_text = question_data['question']
            
            logger.info(f"Processing question {i+1}/{len(questions)}: {question_id}")
            logger.debug(f"Question text: {question_text[:100]}...")
            
            try:
                # Generate response and capture activations
                response, activations = self.inference_client.generate_with_activations(
                    question_text, 
                    capture_layers=self.config.target_layers
                )
                
                # Store activation data
                activation_file = self.activation_storage.save_activation_data(
                    question_id=question_id,
                    question_data=question_data,
                    response=response,
                    activations=activations
                )
                
                logger.info(f"✓ Successfully processed question {question_id}")
                logger.debug(f"Saved to: {activation_file}")
                processed += 1
                
            except Exception as e:
                logger.error(f"✗ Error processing question {question_id}: {e}")
                errors += 1
                continue
        
        return {"processed": processed, "errors": errors}
    
    def list_stored_activations(self) -> List[Dict[str, Any]]:
        """List all stored activation data"""
        return self.activation_storage.list_stored_activations()
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics"""
        return self.activation_storage.get_storage_stats()
    
    def cleanup(self):
        """Clean up resources"""
        try:
            self.question_retriever.close()
            self.inference_client.close()
            logger.info("Cleanup completed")
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Capture model activations for censorship mapping questions"
    )
    
    parser.add_argument(
        "--filter-type", 
        default="all",
        choices=["all", "censored", "uncensored"],
        help="Type of questions to process (default: all)"
    )
    
    parser.add_argument(
        "--max-questions",
        type=int,
        help="Maximum number of questions to process"
    )
    
    parser.add_argument(
        "--target-layers",
        type=str,
        help="Comma-separated list of layer indices (e.g., '5,10,15,20')"
    )
    
    parser.add_argument(
        "--output-format",
        choices=["npz", "h5", "pickle"],
        default="npz",
        help="Output format for activation data (default: npz)"
    )
    
    parser.add_argument(
        "--no-compression",
        action="store_true",
        help="Disable compression for output files"
    )
    
    parser.add_argument(
        "--storage-path",
        help="Path to store activation data"
    )
    
    parser.add_argument(
        "--list-stored",
        action="store_true",
        help="List stored activation files and exit"
    )
    
    parser.add_argument(
        "--storage-stats",
        action="store_true", 
        help="Show storage statistics and exit"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create custom config if needed
    custom_config = None
    if any([args.target_layers, args.output_format, args.no_compression, args.storage_path]):
        custom_config = ActivationCaptureConfig()
        
        if args.target_layers:
            custom_config.target_layers = [int(x.strip()) for x in args.target_layers.split(',')]
        
        if args.output_format:
            custom_config.save_format = args.output_format
        
        if args.no_compression:
            custom_config.compression = False
        
        if args.storage_path:
            custom_config.activations_storage_path = args.storage_path
    
    # Initialize runner
    runner = ActivationCaptureRunner(custom_config)
    
    try:
        # Handle info commands
        if args.list_stored:
            stored_activations = runner.list_stored_activations()
            print(f"\nFound {len(stored_activations)} stored activation files:")
            for item in stored_activations:
                print(f"  {item['timestamp']}: {item['question_id']} - {item['question']}")
            return
        
        if args.storage_stats:
            stats = runner.get_storage_stats()
            print("\nStorage Statistics:")
            for key, value in stats.items():
                print(f"  {key}: {value}")
            return
        
        # Run activation capture
        results = runner.run(
            filter_type=args.filter_type,
            max_questions=args.max_questions,
            target_layers=custom_config.target_layers if custom_config else None
        )
        
        # Print summary
        print(f"\n{'='*50}")
        print("ACTIVATION CAPTURE SUMMARY")
        print(f"{'='*50}")
        print(f"Status: {results['status']}")
        print(f"Total questions: {results['total_questions']}")
        print(f"Successfully processed: {results['processed']}")
        print(f"Errors: {results['errors']}")
        print(f"Duration: {results['duration_seconds']:.2f} seconds")
        print(f"Processing rate: {results['questions_per_second']:.2f} questions/sec")
        print(f"Target layers: {results['config']['target_layers']}")
        print(f"Storage format: {results['config']['save_format']}")
        print(f"Compression: {results['config']['compression']}")
        print(f"Storage size: {results['storage_stats']['total_storage_size_mb']:.2f} MB")
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)
    finally:
        runner.cleanup()

if __name__ == "__main__":
    main()