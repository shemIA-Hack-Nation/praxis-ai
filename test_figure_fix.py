"""
Test the figure inclusion fix
"""
import json

# Test data with sample notebook
test_notebook = """
{
  "cells": [
    {
      "cell_type": "code",
      "source": [
        "import tensorflow as tf\\n",
        "import numpy as np\\n",
        "(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()\\n",
        "x_train, x_test = x_train / 255.0, x_test / 255.0\\n",
        "model = tf.keras.models.Sequential([\\n",
        "  tf.keras.layers.Reshape((28, 28, 1), input_shape=(28, 28)),\\n",
        "  tf.keras.layers.Conv2D(32, 3, activation='relu'),\\n",
        "  tf.keras.layers.MaxPooling2D(),\\n",
        "  tf.keras.layers.Conv2D(64, 3, activation='relu'),\\n",
        "  tf.keras.layers.MaxPooling2D(),\\n",
        "  tf.keras.layers.Flatten(),\\n",
        "  tf.keras.layers.Dense(64, activation='relu'),\\n",
        "  tf.keras.layers.Dense(10)\\n",
        "])\\n",
        "model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])\\n",
        "history = model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))\\n",
        "test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)\\n",
        "print('Test accuracy:', test_acc)"
      ]
    }
  ],
  "metadata": {
    "kernelspec": {
      "display_name": "Python 3",
      "language": "python",
      "name": "python3"
    }
  }
}
"""

def test_figure_inclusion():
    """Test that the orchestrator includes figures in LaTeX"""
    
    print("🧪 Testing Figure Inclusion Fix")
    print("=" * 50)
    
    try:
        # Save test notebook
        with open('test_figures_notebook.ipynb', 'w') as f:
            f.write(test_notebook)
        
        # Import orchestrator
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))
        
        from backend.app.agents.papergen.orchestrator_papergen import PapergenOrchestrator
        from backend.app.agents.agent_tools import parse_notebook
        
        # Parse notebook
        parsed_cells = parse_notebook('test_figures_notebook.ipynb')
        print(f"✅ Parsed notebook with {len(parsed_cells)} cells")
        
        # Create orchestrator
        orchestrator = PapergenOrchestrator()
        print("✅ Created orchestrator")
        
        # Run workflow
        print("🚀 Running paper generation workflow...")
        result = orchestrator.orchestrate_papergen(
            notebook_cells=parsed_cells,
            template_content="Standard research paper template",
            user_requirements={"focus": "CNN image classification"}
        )
        
        if result["success"]:
            print("✅ Workflow completed successfully")
            print(f"📄 LaTeX file: {result.get('latex_file', 'Not generated')}")
            print(f"📊 Illustrations count: {len(result.get('illustrations', []))}")
            
            # Check if LaTeX contains figure references
            latex_file = result.get('latex_file', '')
            if latex_file:
                try:
                    with open(latex_file, 'r', encoding='utf-8') as f:
                        latex_content = f.read()
                    
                    figure_count = latex_content.count('\\includegraphics')
                    if figure_count > 0:
                        print(f"🎉 SUCCESS: Found {figure_count} figures in LaTeX!")
                        print("✅ Figures are now properly included in the research paper")
                        return True
                    else:
                        print("⚠️  LaTeX generated but no figures found")
                        print("📝 Checking for figure placeholders...")
                        placeholder_count = latex_content.count('% Figure:')
                        print(f"Found {placeholder_count} figure placeholders")
                        return False
                        
                except Exception as e:
                    print(f"❌ Error reading LaTeX file: {e}")
                    return False
            else:
                print("❌ No LaTeX file was generated")
                return False
        else:
            print(f"❌ Workflow failed: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
        return False
    
    finally:
        # Cleanup
        try:
            os.remove('test_figures_notebook.ipynb')
        except:
            pass

if __name__ == "__main__":
    success = test_figure_inclusion()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 FIGURE INCLUSION FIX SUCCESSFUL!")
        print("✅ Figures are now being included in research papers")
    else:
        print("⚠️  Issue still needs investigation")
        print("📋 Check if illustrations are being generated and parsed correctly")
    
    exit(0 if success else 1)