# Auto-GUI-Code-Generation

Automatic GUI code generation for web, Android, and iOS, leveraging machine learning and template-based compilers.

## Features

- **Model-driven GUI understanding** via deep learning
- **Template-based code generation** for multiple platforms: Web (HTML), Android (XML), iOS (Storyboard)
- Utilities and data preprocessors for dataset handling

## Directory Overview

```
compiler/           # Platform-specific code generators
  ├── android-compiler.py
  ├── ios-compiler.py
  ├── web-compiler.py
  └── classes/      # Compiler internals (Compiler, Utils, Node)
model/              # ML and data pre-processing
  ├── Extract_features.py
  ├── data_process.py
  └── classes/
      ├── Utils.py
      └── model/
          └── Config.py
test.py             # Example or test script
requirements.txt    # Python dependencies
```

## Getting Started

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **(Optional) Prepare dataset** according to `model/data_process.py` requirements.

3. **Run the model or compilers:**
   - To pre-process data:  
     ```bash
     python model/data_process.py
     ```
   - To extract visual features (requires TensorFlow):  
     ```bash
     python model/Extract_features.py
     ```
   - To generate code for each platform:  
     ```bash
     python compiler/web-compiler.py <input_path>
     python compiler/android-compiler.py <input_file>
     python compiler/ios-compiler.py <input_file>
     ```

## Linting

- **Lint with [Ruff](https://github.com/astral-sh/ruff):**
  ```bash
  pip install ruff
  ruff .
  # To auto-fix issues:
  ruff check . --fix
  ```

- **Recommended:** Add Ruff or your preferred linter as a pre-commit hook and/or to your CI workflow.

## License

[MIT](LICENSE)

## References

- Project code is based on original research and open-source contributions by Tony Beltramelli and others.

---

*See also the linter documentation for advanced configuration and customization: [Ruff Docs](https://docs.astral.sh/ruff/).*