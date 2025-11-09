"""
Illustration Critic Agent (Complete Visualizer)
------------------------------------------------
Role: Analyzes research paper drafts and GENERATES actual images for text-heavy sections.

Input Format:
    Text snippet describing technical concepts without visuals.

Output Format:
    INFO: Found [description]
    TASK: Generated [type] for topic: "[topic]"
    RESULT: [filepath].png
    CAPTION: [LaTeX-ready caption]
"""

import os
import re
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
from pathlib import Path
from datetime import datetime
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure matplotlib for academic quality
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9

# Output directory for generated images
OUTPUT_DIR = Path("generated_figures")
OUTPUT_DIR.mkdir(exist_ok=True)


def get_working_llm():
    """Initialize Gemini LLM."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("❌ GEMINI_API_KEY not found in .env file.")
    
    model_name = os.getenv("GEMINI_MODEL", "models/gemini-2.5-flash")
    
    return ChatGoogleGenerativeAI(
        model=model_name,
        temperature=0.3,
        google_api_key=api_key
    )


# ------------------------------
# Image Generation Functions
# ------------------------------

def generate_flowchart(title: str, steps: list, filename: str) -> str:
    """Generate a flowchart diagram."""
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, len(steps) + 1)
    ax.axis('off')
    
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']
    
    for i, step in enumerate(steps):
        y = len(steps) - i
        color = colors[i % len(colors)]
        
        # Draw box
        box = mpatches.FancyBboxPatch(
            (2, y - 0.3), 6, 0.6,
            boxstyle="round,pad=0.1",
            edgecolor='black',
            facecolor=color,
            alpha=0.7,
            linewidth=2
        )
        ax.add_patch(box)
        
        # Add text
        ax.text(5, y, step, ha='center', va='center', 
                fontsize=10, weight='bold', color='white')
        
        # Add arrow
        if i < len(steps) - 1:
            ax.arrow(5, y - 0.3, 0, -0.3, head_width=0.2, 
                    head_length=0.1, fc='black', ec='black', linewidth=2)
    
    plt.title(title, fontsize=14, weight='bold', pad=20)
    filepath = OUTPUT_DIR / filename
    plt.savefig(filepath, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return str(filepath)


def generate_architecture_diagram(title: str, layers: list, filename: str) -> str:
    """Generate neural network architecture diagram."""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, len(layers) + 1)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    layer_colors = {
        'conv': '#3498db',
        'pool': '#2ecc71',
        'flatten': '#f39c12',
        'dense': '#e74c3c',
        'dropout': '#9b59b6'
    }
    
    for i, layer_info in enumerate(layers):
        x = i + 1
        layer_type = layer_info.get('type', 'dense').lower()
        layer_name = layer_info.get('name', f'Layer {i+1}')
        
        color = layer_colors.get(layer_type, '#95a5a6')
        
        # Draw layer box
        width = 0.6
        height = 3 if 'conv' in layer_type else 2
        y_center = 5
        
        box = mpatches.FancyBboxPatch(
            (x - width/2, y_center - height/2), width, height,
            boxstyle="round,pad=0.05",
            edgecolor='black',
            facecolor=color,
            alpha=0.8,
            linewidth=2
        )
        ax.add_patch(box)
        
        # Add label
        ax.text(x, y_center, layer_name, ha='center', va='center',
                fontsize=9, weight='bold', color='white', rotation=0)
        
        # Add arrow
        if i < len(layers) - 1:
            ax.arrow(x + width/2, y_center, 0.3, 0, 
                    head_width=0.3, head_length=0.1,
                    fc='black', ec='black', linewidth=2)
    
    plt.title(title, fontsize=14, weight='bold', pad=20)
    filepath = OUTPUT_DIR / filename
    plt.savefig(filepath, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return str(filepath)


def generate_comparison_chart(title: str, data: dict, filename: str) -> str:
    """Generate model comparison bar chart."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = list(data.keys())
    values = list(data.values())
    
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    bars = ax.bar(models, values, color=colors[:len(models)], 
                   alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=10, weight='bold')
    
    ax.set_ylabel('Performance Metric', fontsize=12, weight='bold')
    ax.set_xlabel('Model', fontsize=12, weight='bold')
    ax.set_title(title, fontsize=14, weight='bold', pad=20)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    filepath = OUTPUT_DIR / filename
    plt.savefig(filepath, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return str(filepath)


def generate_time_series_plot(title: str, data: dict, filename: str) -> str:
    """Generate time series comparison plot."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for i, (label, values) in enumerate(data.items()):
        color = ['#3498db', '#e74c3c', '#2ecc71'][i % 3]
        ax.plot(values, label=label, linewidth=2.5, 
                marker='o', markersize=5, color=color, alpha=0.8)
    
    ax.set_xlabel('Time Step', fontsize=12, weight='bold')
    ax.set_ylabel('Value', fontsize=12, weight='bold')
    ax.set_title(title, fontsize=14, weight='bold', pad=20)
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    filepath = OUTPUT_DIR / filename
    plt.savefig(filepath, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return str(filepath)


def generate_heatmap(title: str, data: np.ndarray, labels: list, filename: str) -> str:
    """Generate confusion matrix or correlation heatmap."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(data, annot=True, fmt='.2f', cmap='coolwarm',
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Value'}, linewidths=0.5,
                ax=ax, vmin=0, vmax=1)
    
    ax.set_title(title, fontsize=14, weight='bold', pad=20)
    plt.tight_layout()
    filepath = OUTPUT_DIR / filename
    plt.savefig(filepath, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return str(filepath)


# ------------------------------
# Analysis & Generation
# ------------------------------

ANALYSIS_PROMPT = """
You are the **IllustrationCritic Agent**, a specialized data extraction and visualization analyst for research papers.

Your task: Read the draft, extract EXACT data points, and create structured visualization specifications.

For EACH section needing visualization, return a structured request with PRECISE data extraction:

VISUALIZATION_REQUEST
TYPE: [flowchart|architecture|comparison|timeseries|heatmap]
TOPIC: [brief description of what to visualize]
DETAILS: [specific structure/flow to show]
DATA_EXTRACTION: [CRITICAL - Extract exact numbers, labels, and values from the text]
AXIS_LABELS: X=[label], Y=[label], TITLE=[title]
LABELS: [comma-separated list of all categorical labels/names]
VALUES: [comma-separated list of all numeric values in same order as LABELS]
FILENAME: [descriptive_name_in_snake_case]
END_REQUEST

**CRITICAL RULES FOR DATA EXTRACTION:**
1. For COMPARISON charts: Extract ALL model names and their EXACT metric values
2. For TIMESERIES: Extract time points (weeks/days/epochs) and corresponding values for EACH series
3. For HEATMAP: Extract matrix dimensions, row labels, column labels, and cell values
4. For ARCHITECTURE: Extract layer sequence with parameters (filters, neurons, dimensions)
5. For FLOWCHART: Extract sequential steps in exact order

**DATA_EXTRACTION FORMAT BY TYPE:**

COMPARISON:
DATA_EXTRACTION: Models: [Model1, Model2, Model3], Metric: MAE, Values: [0.045, 0.062, 0.078]

TIMESERIES:
DATA_EXTRACTION: TimePoints: [1,2,3,4,5,6,7,8,9,10], Series1_Name: [v1,v2,v3,...], Series2_Name: [v1,v2,v3,...]

HEATMAP:
DATA_EXTRACTION: Rows: [Var1,Var2,Var3], Columns: [Var1,Var2,Var3], Matrix: [[1.0,0.5,0.3],[0.5,1.0,0.7],[0.3,0.7,1.0]]

ARCHITECTURE:
DATA_EXTRACTION: Layers: [Conv2D(32,3x3), MaxPool(2x2), Conv2D(64,3x3), Flatten, Dense(128), Dense(10)]

FLOWCHART:
DATA_EXTRACTION: Steps: [Step1, Step2, Step3, Step4, Step5]

EXAMPLE OUTPUT:

VISUALIZATION_REQUEST
TYPE: comparison
TOPIC: Machine Learning Model Performance Comparison
DETAILS: Bar chart comparing MAE scores across 5 forecasting models
DATA_EXTRACTION: Models: [LSTM, Prophet, ARIMA, Random Forest, Linear Regression], Metric: MAE, Values: [0.0423, 0.0618, 0.0721, 0.0545, 0.0892]
AXIS_LABELS: X=Model Name, Y=Mean Absolute Error (MAE), TITLE=Forecasting Model Performance Comparison
LABELS: LSTM, Prophet, ARIMA, Random Forest, Linear Regression
VALUES: 0.0423, 0.0618, 0.0721, 0.0545, 0.0892
FILENAME: ml_model_mae_comparison
END_REQUEST

VISUALIZATION_REQUEST
TYPE: timeseries
TOPIC: Weekly pharmaceutical demand forecast vs actual
DETAILS: Multi-line plot showing actual demand and predictions from two models over 13 weeks
DATA_EXTRACTION: TimePoints: [1,2,3,4,5,6,7,8,9,10,11,12,13], Actual: [1234,1456,1823,2145,1987,1756,1645,1534,1423,1678,1890,2012,1845], LSTM: [1189,1478,1801,2178,2012,1723,1667,1556,1398,1701,1867,2034,1823], Prophet: [1156,1402,1689,1945,2234,1856,1734,1645,1512,1589,1756,1923,1867]
AXIS_LABELS: X=Week Number, Y=Demand (Units), TITLE=Drug A Weekly Demand: Actual vs Predicted
LABELS: Week 1, Week 2, Week 3, Week 4, Week 5, Week 6, Week 7, Week 8, Week 9, Week 10, Week 11, Week 12, Week 13
VALUES: Actual, LSTM, Prophet
FILENAME: drug_a_demand_timeseries
END_REQUEST

VISUALIZATION_REQUEST
TYPE: heatmap
TOPIC: Supply chain variable correlation matrix
DETAILS: 5x5 correlation heatmap showing relationships between operational variables
DATA_EXTRACTION: Rows: [Demand, Inventory, LeadTime, Cost, Satisfaction], Columns: [Demand, Inventory, LeadTime, Cost, Satisfaction], Matrix: [[1.000,0.567,-0.234,0.423,0.312],[0.567,1.000,-0.734,0.156,-0.089],[-0.234,-0.734,1.000,-0.456,-0.678],[0.423,0.156,-0.456,1.000,-0.234],[0.312,-0.089,-0.678,-0.234,1.000]]
AXIS_LABELS: X=Variables, Y=Variables, TITLE=Supply Chain Correlation Matrix
LABELS: Demand, Inventory, LeadTime, Cost, Satisfaction
VALUES: correlation_coefficients
FILENAME: supply_chain_correlation_heatmap
END_REQUEST

VISUALIZATION_REQUEST
TYPE: architecture
TOPIC: CNN model architecture for medical image classification
DETAILS: Sequential diagram showing layer progression with dimensions
DATA_EXTRACTION: Layers: [Input(224x224x3), Conv2D(32 filters, 3x3), BatchNorm, MaxPool(2x2), Conv2D(64 filters, 3x3), BatchNorm, MaxPool(2x2), Conv2D(128 filters, 3x3), MaxPool(2x2), Flatten(86528), Dense(256), Dropout(0.5), Dense(128), Dense(10)]
AXIS_LABELS: X=Layer Sequence, Y=Not Applicable, TITLE=CNN Architecture for Medical Diagnosis
LABELS: Input, Conv2D-32, BN, MaxPool, Conv2D-64, BN, MaxPool, Conv2D-128, MaxPool, Flatten, Dense-256, Dropout, Dense-128, Output-10
VALUES: layer_parameters
FILENAME: cnn_medical_architecture
END_REQUEST

Now analyze this draft and generate 2-4 visualization requests with EXACT data extraction:

{paper_draft}

**IMPORTANT:** 
- Extract ONLY data that EXISTS in the text (no invention)
- Use exact numerical values as written
- Preserve order and labels exactly
- If data is incomplete, note it in DATA_EXTRACTION
- Return ONLY VISUALIZATION_REQUEST blocks, no commentary
"""


def parse_visualization_requests(response: str) -> list:
    """Parse LLM response into structured visualization requests with data extraction."""
    requests = []
    blocks = response.split('VISUALIZATION_REQUEST')
    
    for block in blocks[1:]:  # Skip first empty split
        if 'END_REQUEST' not in block:
            continue
            
        lines = block.strip().split('\n')
        request = {
            'labels': [],
            'values': [],
            'data_points': {}
        }
        
        for line in lines:
            if ':' not in line:
                continue
                
            key, value = line.split(':', 1)
            key = key.strip().lower()
            value = value.strip()
            
            if key == 'type':
                request['type'] = value
            elif key == 'topic':
                request['topic'] = value
            elif key == 'details':
                request['details'] = value
            elif key == 'filename':
                request['filename'] = value
            elif key == 'data_extraction':
                request['data_extraction'] = value
                # Parse data extraction into usable format
                request['data_points'] = parse_data_extraction(value, request.get('type', ''))
            elif key == 'axis_labels':
                request['axis_labels'] = value
            elif key == 'labels':
                # Split by comma and clean
                request['labels'] = [label.strip() for label in value.split(',') if label.strip()]
            elif key == 'values':
                # Try to parse as numbers, keep as strings if not
                value_list = [v.strip() for v in value.split(',') if v.strip()]
                try:
                    request['values'] = [float(v) for v in value_list]
                except ValueError:
                    request['values'] = value_list
        
        if 'type' in request and 'topic' in request:
            requests.append(request)
    
    return requests


def parse_data_extraction(data_str: str, viz_type: str) -> dict:
    """
    Parse DATA_EXTRACTION string into structured dictionary based on visualization type.
    """
    data = {}
    
    try:
        # Split by known patterns
        if 'Models:' in data_str and 'Values:' in data_str:
            # Comparison type
            models_match = re.search(r'Models:\s*\[(.*?)\]', data_str)
            values_match = re.search(r'Values:\s*\[(.*?)\]', data_str)
            
            if models_match and values_match:
                data['labels'] = [m.strip() for m in models_match.group(1).split(',')]
                data['values'] = [float(v.strip()) for v in values_match.group(1).split(',')]
        
        elif 'TimePoints:' in data_str:
            # Timeseries type
            timepoints_match = re.search(r'TimePoints:\s*\[(.*?)\]', data_str)
            if timepoints_match:
                data['timepoints'] = [int(t.strip()) for t in timepoints_match.group(1).split(',')]
            
            # Extract all series
            series_pattern = r'(\w+):\s*\[([\d,.\s]+)\]'
            for match in re.finditer(series_pattern, data_str):
                series_name = match.group(1)
                if series_name != 'TimePoints':
                    values = [float(v.strip()) for v in match.group(2).split(',')]
                    data[series_name] = values
        
        elif 'Matrix:' in data_str:
            # Heatmap type
            rows_match = re.search(r'Rows:\s*\[(.*?)\]', data_str)
            cols_match = re.search(r'Columns:\s*\[(.*?)\]', data_str)
            matrix_match = re.search(r'Matrix:\s*\[(.*)\]', data_str)
            
            if rows_match:
                data['row_labels'] = [r.strip() for r in rows_match.group(1).split(',')]
            if cols_match:
                data['col_labels'] = [c.strip() for c in cols_match.group(1).split(',')]
            if matrix_match:
                # Parse nested list structure
                matrix_str = matrix_match.group(1)
                rows = re.findall(r'\[([\d,.\s-]+)\]', matrix_str)
                data['matrix'] = [[float(v.strip()) for v in row.split(',')] for row in rows]
        
        elif 'Layers:' in data_str:
            # Architecture type
            layers_match = re.search(r'Layers:\s*\[(.*)\]', data_str)
            if layers_match:
                data['layers'] = [l.strip() for l in layers_match.group(1).split(',')]
        
        elif 'Steps:' in data_str:
            # Flowchart type
            steps_match = re.search(r'Steps:\s*\[(.*)\]', data_str)
            if steps_match:
                data['steps'] = [s.strip() for s in steps_match.group(1).split(',')]
    
    except Exception as e:
        print(f"⚠️ Warning: Could not parse data extraction: {e}")
    
    return data


def generate_visualization(request: dict) -> tuple:
    """Generate visualization based on request type with extracted data."""
    viz_type = request.get('type', 'flowchart')
    topic = request.get('topic', 'Visualization')
    details = request.get('details', '')
    filename = request.get('filename', f'figure_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    data_points = request.get('data_points', {})
    labels = request.get('labels', [])
    values = request.get('values', [])
    
    if not filename.endswith('.png'):
        filename += '.png'
    
    filepath = None
    
    try:
        if viz_type == 'flowchart':
            # Use extracted steps or parse from details
            steps = data_points.get('steps') or [s.strip() for s in re.split(r'[→->]', details) if s.strip()]
            if not steps and labels:
                steps = labels
            if not steps:
                steps = ['Initialize', 'Process', 'Evaluate', 'Iterate', 'Output']
            filepath = generate_flowchart(topic, steps, filename)
            
        elif viz_type == 'architecture':
            # Use extracted layers
            layer_data = data_points.get('layers', [])
            if not layer_data and labels:
                layer_data = labels
            
            layers = []
            for layer_info in layer_data:
                # Parse layer name and type
                if isinstance(layer_info, str):
                    layer_match = re.match(r'(\w+)\(?(.*?)\)?', layer_info)
                    if layer_match:
                        layer_name = layer_match.group(1)
                        params = layer_match.group(2)
                        layers.append({
                            'type': layer_name.lower(),
                            'name': f"{layer_name}\n{params}" if params else layer_name
                        })
            
            if not layers:
                layers = [
                    {'type': 'conv', 'name': 'Conv2D'},
                    {'type': 'pool', 'name': 'MaxPool'},
                    {'type': 'dense', 'name': 'Dense'}
                ]
            filepath = generate_architecture_diagram(topic, layers, filename)
            
        elif viz_type == 'comparison':
            # Use extracted data
            comp_labels = data_points.get('labels', labels)
            comp_values = data_points.get('values', values)
            
            if comp_labels and comp_values and len(comp_labels) == len(comp_values):
                data = dict(zip(comp_labels, comp_values))
            else:
                # Fallback to sample data
                data = {'Model A': 0.85, 'Model B': 0.72, 'Model C': 0.91}
            
            filepath = generate_comparison_chart(topic, data, filename)
            
        elif viz_type == 'timeseries':
            # Use extracted time series data
            timepoints = data_points.get('timepoints', [])
            
            # Extract all series (keys that aren't 'timepoints')
            series_data = {}
            for key, vals in data_points.items():
                if key not in ['timepoints', 'labels', 'values'] and isinstance(vals, list):
                    series_data[key] = vals
            
            if not series_data and labels and values:
                # Fallback: try to reconstruct from labels/values
                if len(values) > len(labels):
                    # Values might be concatenated series
                    series_data = {'Series': values}
            
            if not series_data:
                # Generate sample data
                x = np.arange(10)
                series_data = {
                    'Actual': np.sin(x/2) + np.random.normal(0, 0.1, 10),
                    'Predicted': np.sin(x/2)
                }
            
            filepath = generate_time_series_plot(topic, series_data, filename)
            
        elif viz_type == 'heatmap':
            # Use extracted matrix data
            matrix = data_points.get('matrix', [])
            row_labels = data_points.get('row_labels', labels)
            col_labels = data_points.get('col_labels', labels)
            
            if not matrix or not row_labels:
                # Generate sample correlation matrix
                size = len(row_labels) if row_labels else 5
                matrix = np.random.uniform(0, 1, (size, size))
                matrix = (matrix + matrix.T) / 2  # Make symmetric
                np.fill_diagonal(matrix, 1)
                if not row_labels:
                    row_labels = [f'Var{i+1}' for i in range(size)]
                    col_labels = row_labels
            
            matrix_array = np.array(matrix) if not isinstance(matrix, np.ndarray) else matrix
            filepath = generate_heatmap(topic, matrix_array, row_labels, filename)
            
    except Exception as e:
        print(f"⚠️ Error generating {viz_type}: {e}")
        return None, None
    
    # Generate detailed caption with axis information
    axis_info = request.get('axis_labels', '')
    caption = f"Figure: {topic}. "
    
    if axis_info:
        caption += f"{axis_info}. "
    
    if viz_type == 'comparison' and labels and values:
        caption += f"Comparing {len(labels)} models. "
    elif viz_type == 'timeseries' and data_points:
        n_series = len([k for k in data_points.keys() if k not in ['timepoints', 'labels', 'values']])
        caption += f"Time series with {n_series} series. "
    elif viz_type == 'heatmap' and labels:
        caption += f"{len(labels)}×{len(labels)} correlation matrix. "
    
    caption += details[:100] + ('...' if len(details) > 100 else '')
    
    return filepath, caption


def agent_illustration_critic(paper_draft: str) -> str:
    """
    Complete visualization pipeline: analyze draft and generate images.
    """
    try:
        llm = get_working_llm()
        
        # Step 1: Analyze draft
        prompt = ChatPromptTemplate.from_template(ANALYSIS_PROMPT)
        formatted_prompt = prompt.format_messages(paper_draft=paper_draft)
        response = llm.invoke(formatted_prompt)
        
        # Step 2: Parse requests
        requests = parse_visualization_requests(response.content)
        
        if not requests:
            return "INFO: No visualization needs identified in this draft section."
        
        # Step 3: Generate images
        results = []
        for i, request in enumerate(requests, 1):
            filepath, caption = generate_visualization(request)
            
            if filepath:
                results.append(
                    f"INFO: Found text describing {request.get('topic', 'concept')} without visual support.\n"
                    f"TASK: Generated {request.get('type', 'diagram')} for topic: \"{request.get('topic', 'N/A')}\"\n"
                    f"RESULT: {filepath}\n"
                    f"CAPTION: {caption}\n"
                )
        
        return "\n".join(results) if results else "INFO: Image generation encountered errors."
        
    except Exception as e:
        return f"❌ Error executing IllustrationCritic Agent: {str(e)}"


# ------------------------------
# Test Cases
# ------------------------------
if __name__ == "__main__":
    print("🎨 Running IllustrationCritic Agent (Complete Visualizer)...\n")
    print("=" * 70)
    
    # Test Case 1: Genetic Algorithm
    test_ga = """
    In this study, we implemented a Genetic Algorithm (GA) to optimize neural network 
    hyperparameters. The GA generates an initial population of random solutions, where 
    each individual represents hyperparameters encoded as a chromosome. During each 
    generation, individuals are evaluated using a fitness function measuring model 
    accuracy. Top performers undergo crossover and mutation to produce offspring. 
    The algorithm iterates until convergence. Results show the GA consistently discovers 
    better configurations than random search or grid search, with a 15% improvement in 
    final model accuracy.
    """
    
    # Test Case 2: CNN Architecture
    test_cnn = """
    The model architecture begins with a 2D Convolutional layer (32 filters, 3x3 kernel), 
    followed by MaxPooling (2x2), another Conv2D layer (64 filters), MaxPooling, a Flatten 
    layer, and two Dense layers (128 neurons with ReLU, then 10 neurons with softmax). 
    The model achieved 94% accuracy on MNIST. Dropout (0.5) was applied after the first 
    Dense layer to prevent overfitting.
    """
    
    # Test Case 3: LSTM Performance
    test_lstm = """
    The LSTM model demonstrated superior forecasting performance with MAE of 0.045, 
    significantly lower than Prophet's 0.062. We compared predicted versus actual demand 
    for three pharmaceutical products during Q3 2023. The LSTM showed closer adherence 
    to actual demand, especially during high-volatility periods. Training used 80% of 
    historical data with Adam optimizer and learning rate 0.001.
    """
    
    test_cases = [
        ("Genetic Algorithm", test_ga),
        ("CNN Architecture", test_cnn),
        ("LSTM Forecasting", test_lstm)
    ]
    
    for test_name, test_input in test_cases:
        print(f"\n📝 TEST: {test_name}")
        print("-" * 70)
        result = agent_illustration_critic(test_input)
        print(result)
        print()
    
    print("=" * 70)
    print(f"✅ Testing complete. Check '{OUTPUT_DIR}' for generated images.")