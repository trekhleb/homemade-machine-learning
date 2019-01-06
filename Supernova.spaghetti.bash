Skip to content
 
Search or jump to…

Pull requests
Issues
Marketplace
Explore
 @therealvenenoiot Sign out
0
1 97 therealvenenoiot/codecov-bash
forked from codecov/codecov-bash
 Code  Pull requests 0  Projects 0  Insights  Settings
Global coverage report uploader for Codecov https://codecov.io
 665 commits
 9 branches
 1 release
 28 contributors
 Apache-2.0
 Shell 99.6%	 Other 0.4%
 Pull request   Compare This branch is 1 commit ahead of codecov:master.
@therealvenenoiot
therealvenenoiot Create CyberWorld.git
Latest commit e762e89  2 minutes ago
Type	Name	Latest commit message	Commit time
ignores	fix kt blocks	2 years ago
tests	update drone integration	a year ago
.gitignore	fix tests	2 years ago
.travis.yml	fix tests	2 years ago
CyberWorld.git	Create CyberWorld.git	2 minutes ago
LICENSE	Create LICENSE	2 years ago
circle.yml	fix tests	2 years ago
codecov	suppress gcov output via -- codecov#121	8 months ago
env	update drone integration	a year ago
readme.md	update readme.md close codecov#112	a year ago
shunit2-2.1.6.tgz	fix tests	2 years ago
 readme.md
Codecov Global Uploader
Upload reports to Codecov for almost every supported language.

Deployed Version

# All CI
bash <(curl -s https://codecov.io/bash)

# Pipe to bash (Jenkins)
curl -s https://codecov.io/bash | bash -s - -t token
                                            ^ add your extra config here

# No bash method
curl -s https://codecov.io/bash > .codecov
chmod +x .codecov
./.codecov
Languages
Python, C#/.net, Java, Node/Javascript/Coffee, C/C++, D, Go, Groovy, Kotlin, PHP, R, Scala, Xtern, Xcode, Lua and more...

Usage
Below are most commonly used settings. View full list of commands to see the full list of commands.

# public repo on Travis CI
after_success:
  - bash <(curl -s https://codecov.io/bash)
# private repo
after_success:
  - bash <(curl -s https://codecov.io/bash) -t your-repository-upload-token
# Flag build types
after_success:
  - bash <(curl -s https://codecov.io/bash) -F unittests
# Include environment variables to store per build
after_success:
  - bash <(curl -s https://codecov.io/bash) -e TOX_ENV,CUSTOM_VAR
Prevent build failures
If Codecov fails to upload reports, you can ensure the CI build does not fail by adding a catch-all:

bash <(curl -s https://codecov.io/bash) || echo "Codecov did not collect coverage reports"
CI Providers
Company	Supported	Token Required
Travis CI	Yes Build Status	Private only
CircleCI	Yes Circle CI	Private only
Codeship	Yes	Public & Private
Jenkins	Yes	Public & Private
Semaphore	Yes	Public & Private
drone.io	Yes	Public & Private
AppVeyor	No. See Codecov Python.	Private only
Wercker	Yes	Public & Private
Magnum CI	Yes	Public & Private
Shippable	Yes	Public & Private
Gitlab CI	Yes	Public & Private
git	Yes (as a fallback)	Public & Private
Buildbot	coming soon buildbot/buildbot#1671	
Bamboo	coming soon	
Solano Labs	coming soon	
Bitrise	coming soon	
Using Travis CI? Uploader is compatible with sudo: false which can speed up your builds. +1

Caveat
Jenkins: Unable to find reports? Try PWD=WORKSPACE bash <(curl -s https://codecov.io/bash)
Please let us know if this works for you. More at #112
© 2019 GitHub, Inc.
Terms
Privacy
Security
Status
Help
Contact GitHub
Pricing
API
Training
Blog
About
Press h to open a hovercard with more details.
