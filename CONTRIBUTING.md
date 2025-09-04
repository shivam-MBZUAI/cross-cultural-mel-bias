# Contributing to Cross-Cultural Mel-Scale Audio Bias Research

Thank you for your interest in contributing to our research on cross-cultural bias in audio representations!

## Current Status

🚧 **This repository currently contains only the dataset downloader.** The complete analysis code, models, and experiments will be released upon paper acceptance.

## How to Contribute (Dataset Downloader Phase)

### Bug Reports
If you encounter issues with the dataset downloader:

1. **Check existing issues** first to avoid duplicates
2. **Provide detailed information**:
   - Python version and OS
   - Full error message and stack trace
   - Dataset you were trying to download
   - Authentication setup details (without sharing tokens!)

### Feature Requests
We welcome suggestions for:
- Additional dataset support
- Improved authentication handling
- Better error messages and user experience
- Cross-platform compatibility improvements

### Code Contributions
1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/your-feature-name`
3. **Test your changes**: Ensure the downloader works with your modifications
4. **Follow code style**: Use black for formatting (`black download_datasets.py`)
5. **Update documentation**: Update README.md if needed
6. **Submit a pull request**

### Development Setup
```bash
git clone https://github.com/shivam-MBZUAI/cross-cultural-mel-bias.git
cd cross-cultural-mel-bias
pip install -r requirements.txt

# Test the downloader
python download_datasets.py --list
```

## Future Contributions (After Full Release)

Once the complete framework is released, we'll welcome contributions in:
- Audio front-end implementations
- Cross-cultural bias metrics
- New evaluation protocols
- Documentation improvements
- Reproducibility enhancements

## Code of Conduct

- **Be respectful** in all interactions
- **Focus on constructive feedback**
- **Respect intellectual property** and cite sources appropriately
- **Follow ethical guidelines** for dataset usage and research

## Questions?

Feel free to open an issue for:
- Technical questions about the downloader
- Clarifications about the research methodology
- Suggestions for improvement

**Contact**: [0shivam33@gmail.com](mailto:0shivam33@gmail.com)

---

🌍 **Building Fair Audio AI for Global Diversity**
