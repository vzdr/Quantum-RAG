"""Test script to verify the service works with new code."""
import sys
from pathlib import Path

# Add parent paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

print("Testing service initialization...")

try:
    from demo_app.backend.services.retrieval_service import get_retrieval_service

    print("✓ Import successful")

    service = get_retrieval_service()
    print("✓ Service created")

    # This will trigger the embedder lazy loading
    embedder = service.embedder
    print("✓ Embedder loaded successfully!")
    print(f"  Model: {embedder.model}")
    print(f"  Device: {embedder.device}")

    print("\n✓✓✓ ALL TESTS PASSED - Code is working correctly!")

except Exception as e:
    print(f"\n✗✗✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
