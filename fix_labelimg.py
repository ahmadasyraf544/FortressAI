import os
import labelImg

pkg_dir = os.path.dirname(labelImg.__file__)
main_file = os.path.join(pkg_dir, 'labelImg.py')

with open(main_file, 'r', encoding='utf-8') as f:
    content = f.read()

# Backup first
backup_file = main_file + ".bak"
if not os.path.exists(backup_file):
    with open(backup_file, 'w', encoding='utf-8') as f_bak:
        f_bak.write(content)
    print(f"Backup created: {backup_file}")

# Fix 1: in add_label, protect against shape is None
old_add_label = """    def add_label(self, shape):
        shape.paint_label = self.display_label_option.isChecked()"""

new_add_label = """    def add_label(self, shape):
        if shape is None:
            return
        shape.paint_label = self.display_label_option.isChecked()"""

# Fix 2: in copy_selected_shape, check shape
old_copy = """    def copy_selected_shape(self):
        self.add_label(self.canvas.copy_selected_shape())
        # fix copy and delete
        self.shape_selection_changed(True)"""

new_copy = """    def copy_selected_shape(self):
        shape = self.canvas.copy_selected_shape()
        if shape is not None:
            self.add_label(shape)
            # fix copy and delete
            self.shape_selection_changed(True)"""

patched = False
if old_add_label in content:
    content = content.replace(old_add_label, new_add_label)
    patched = True

if old_copy in content:
    content = content.replace(old_copy, new_copy)
    patched = True

if patched:
    with open(main_file, 'w', encoding='utf-8') as f:
        f.write(content)
    print("SUCCESS: labelImg.py has been successfully patched!")
else:
    print("WARNING: Patterns not matched or already patched.")
