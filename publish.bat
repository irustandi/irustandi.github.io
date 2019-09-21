git add .
git commit -a -m %1
git push -u origin source
pelican content -o output -s pelicanconf.py
ghp-import output -r origin -b master
git push origin master
git checkout source