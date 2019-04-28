## Contributing

### General Rules

- As much as possible, try to follow the existing format of markdown and code.
- Don't forget to run `pylint ./homemade` before submitting pull requests.

### Contributing New Translation

- Create new `README.xx-XX.md` file with translation alongside with main `README.md` file where `xx-XX` is [locale and country/region codes](http://www.lingoes.net/en/translator/langcode.htm). For example `en-US`, `zh-CN`, `zh-TW`, `ko-KR` etc.
- You may also translate all other sub-folders by creating related `README.xx-XX.md` files in each of them.

### Contributing New Algorithms

- Make your pull requests to be **specific** and **focused**. Instead of contributing "several algorithms" all at once contribute them all one by one separately (i.e. one pull request for "Logistic Regression", another one for "K-Means" and so on).
- Every new algorithm must have:
    - **Source code** with comments and readable namings
    - **Math** being explained in README.md along with the code
    - **Jupyter demo notebook** with example of how this new algorithm may be applied

If you're adding new **datasets** they need to be saved in the `/data` folder. CSV files are preferable. The size of the file should not be greater than `30Mb`.
