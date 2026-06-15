import ast

path = "/mnt/ssd8/bioactive/defaults/trainer.py"
s = open(path).read()

old1 = """            if self.val_target > self.best_val_target:
                self.best_val_target = self.val_target
                if self.save_best_model:
                    self.best_model = model_to_CPU_state(self.model)"""
new1 = """            if self.val_target > self.best_val_target:
                self.best_val_target = self.val_target
                self.patience_counter = 0
                if self.save_best_model:
                    self.best_model = model_to_CPU_state(self.model)
            else:
                self.patience_counter = getattr(self, "patience_counter", 0) + 1
                if getattr(self, "early_stopping", False) and self.patience_counter >= getattr(self, "early_stopping_patience", 8):
                    self.should_stop = True
                    print("EARLY STOPPING: no val improvement for %d checks. Best=%.4f" % (self.patience_counter, self.best_val_target))"""
assert old1 in s, "Edit1 pattern not found"
s = s.replace(old1, new1)

old2 = """                        if self.is_rank0:
                            self._log_epoch_summary()
                        self.model.train()
                synchronize()"""
new2 = """                        if self.is_rank0:
                            self._log_epoch_summary()
                        self.model.train()
                        if getattr(self, "should_stop", False):
                            break
                synchronize()"""
assert old2 in s, "Edit2 pattern not found"
s = s.replace(old2, new2)

old3 = """            if not self.save_best_model and not self.is_grid_search:
                self.best_model = model_to_CPU_state(self.model)   
                self.save_session()"""
new3 = """            if not self.save_best_model and not self.is_grid_search:
                self.best_model = model_to_CPU_state(self.model)   
                self.save_session()
            if getattr(self, "should_stop", False):
                print("EARLY STOPPING: exiting at epoch %d" % self.epoch)
                break"""
assert old3 in s, "Edit3 pattern not found"
s = s.replace(old3, new3)

open(path, "w").write(s)
ast.parse(open(path).read())
print("All 3 edits applied. trainer.py parses OK.")
