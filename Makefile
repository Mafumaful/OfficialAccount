.PHONY: run

run:
	python3 scripts/run_menu.py

debug:
	python3 scripts/run_menu.py --debug

# clean all media file
clean:
	@echo "Cleaning media files..."
	@find . -type d -iname '*media*' -exec rm -rf {} +