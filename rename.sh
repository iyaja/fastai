cd docs
rename 's/.html/.md/' *
mkdir tutorial data vision text tabular callback training core medical
rename 's/\./\//' *.md
mv docs/index.md docs/README.md
mv optimizer.md metrics.md interpret.md distributed.md training/
mv *.md tutorial 
