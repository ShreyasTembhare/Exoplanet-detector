import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Callable, Any

class ParallelProcessor:
    """Parallel processing manager for batch operations."""
    
    def __init__(self, max_workers=None):
        self.max_workers = max_workers
        self.logger = logging.getLogger(__name__)
    
    def process_batch(self, items, process_func, chunk_size=None):
        """Process items in parallel using ThreadPoolExecutor."""
        results = []
        errors = []
        
        try:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all tasks
                future_to_item = {executor.submit(process_func, item): item for item in items}
                
                # Collect results as they complete
                for future in as_completed(future_to_item):
                    item = future_to_item[future]
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        self.logger.error(f"Error processing item {item}: {e}")
                        errors.append((item, str(e)))
            
            self.logger.info(f"Processed {len(results)} items successfully, {len(errors)} errors")
            return results, errors
            
        except Exception as e:
            self.logger.error(f"Parallel processing failed: {e}")
            return [], [(item, str(e)) for item in items] 