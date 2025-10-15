# Multi-Task-Deep-Learning-for-Long-Tailed-X-Ray-Absorption-Studies-of-Aluminum-Alloys
This study uses X-ray imaging and multi-task CNNs to analyze five aluminum alloys across different thicknesses. A tailored imbalance-aware method addresses long-tailed data, enabling accurate alloy classification and absorption analysis, advancing AI-driven material characterization.

This project analyzes five aluminum-based alloys: **Al 1050 (pure aluminum), Al 2017 (Al–Cu), Al 5083 (Al–Mg), Al 6082 (Al–Si–Mg), and Al 7075 (Al–Zn–Mg)**. We use a multi-task CNN framework to simultaneously classify alloy type and predict thickness-dependent absorption. To handle challenges from long-tailed data distributions, an imbalance-aware training method was applied, improving both robustness and generalization of the model.

## Model Hyperparameters

### Hyperparameter Descriptions

| Hyperparameter | Description |
| :--- | :--- |
| **Optimizer** | PBT enables the selection and evolution of optimizers throughout training, dynamically switching between algorithms such as **Adam**, **SGD**, or **RMSprop**. The choice of optimizer influences the convergence speed and stability of the model. |
| **Activation Function** | Activation functions introduce **non-linearity** into the model, enabling it to learn complex patterns. Commonly used activation functions in ResNet include **ReLU**, which mitigates the vanishing gradient problem. |
| **Learning Rate** | The **learning rate** is a critical hyperparameter that controls the step size for weight updates during back-propagation. |
| **Weight Decay** | **Weight decay** (**L2 regularization**) prevents overfitting by penalizing large weight magnitudes. |

### Model Configurations

| Model | Optimizer | Activation Function | Learning Rate | Weight Decay |
| :--- | :--- | :--- | :--- | :--- |
| **ResNet18** | Adam | ReLU | 0.001 | 0.0001 |
| **ResNet34** | SGD | Leaky ReLU | 0.01 | 0.0001 |
| **ResNet50** | RMSprop | GELU | 0.0005 | 0.0005 |
| **FNN** | SGD | ReLU | 0.01 | 0.0001 |
| **VGG16** | Adam / SGD+Momentum | ReLU | 0.0002 | 0.0001 |
| **EfficientNetB0** | RMSprop / Adam | Swish (SiLU) | 0.002 | 0.0001 |
