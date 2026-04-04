# Auto-GUI-Code-Generation

Automatic GUI code generation for web, Android, and iOS, leveraging machine learning and template-based compilers. Based on the [pix2code](https://github.com/tonybeltramelli/pix2code) research by Tony Beltramelli.

The pipeline is: **Screenshot Image --> ML Model --> DSL tokens --> Platform Compiler --> Native Code**

## Features

- **Model-driven GUI understanding** via deep learning (ResNet50 feature extraction)
- **Template-based code generation** for multiple platforms: Web (HTML/Bootstrap), Android (XML), iOS (Storyboard)
- Data preprocessors for dataset handling and train/val/test splitting

## Directory Overview

```
compiler/               # Platform-specific code generators
  ├── android-compiler.py
  ├── ios-compiler.py
  ├── web-compiler.py
  ├── assets/           # DSL-to-platform mapping JSON files
  └── classes/          # Compiler internals (Compiler, Node, Utils)
model/                  # ML and data pre-processing
  ├── Extract_features.py   # ResNet50 feature extraction
  ├── data_process.py       # Dataset preprocessing and splitting
  └── classes/
      ├── Utils.py          # Image preprocessing utilities
      └── model/
          └── Config.py     # Hyperparameter configuration
requirements.txt        # Python dependencies (legacy, see note below)
```

## Getting Started

1. **Install dependencies:**

   > **Note:** The dependencies in `requirements.txt` are pinned to legacy versions (TensorFlow 1.7, Keras 2.1, etc.) from 2017-2018. They require Python 3.6 and may not install on modern systems.

   ```bash
   pip install -r requirements.txt
   ```

2. **(Optional) Prepare dataset** according to `model/data_process.py` requirements. Datasets should be placed at `../datasets/<platform>/all_data/` relative to the `model/` directory.

3. **Run the compilers** (must be run from the `compiler/` directory):
   ```bash
   cd compiler
   python web-compiler.py <path_to_gui_file>
   python android-compiler.py <path_to_gui_file>
   python ios-compiler.py <path_to_gui_file>
   ```

4. **Run data preprocessing:**
   ```bash
   cd model
   python data_process.py
   ```

## Known Limitations

- The ML model training pipeline is incomplete -- only data preprocessing and feature extraction are implemented. There is no training loop or inference script.
- Feature extraction (`Extract_features.py`) uses TensorFlow 1.x APIs (`tf.enable_eager_execution()`) which are deprecated in TensorFlow 2.x.
- The compilers use relative paths to load DSL mapping files from `assets/`, so they must be run from within the `compiler/` directory.

## License

[Apache 2.0](LICENSE)

## References

- Tony Beltramelli, [pix2code: Generating Code from a Graphical User Interface Screenshot](https://github.com/tonybeltramelli/pix2code), 2018.
