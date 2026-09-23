"""Bounded cache of immutable question token sequences; page states are never cached."""
from collections import OrderedDict
import threading

class QuestionTokenCache:
    def __init__(self, tokenize, max_entries=16, max_tokens=8192, max_chars=65536):
        self.tokenize = tokenize
        self.max_entries, self.max_tokens, self.max_chars = max_entries, max_tokens, max_chars
        self.entries = OrderedDict()
        self.lock = threading.Lock()
        self.hits = self.misses = 0

    def get(self, tokenizer, instruction, options):
        options = tuple(options)
        # Avoid retaining oversized or ultimately rejected user input.
        eligible = len(instruction) + sum(map(len, options)) <= self.max_chars
        key = (id(tokenizer), instruction, options)
        with self.lock:
            if eligible and key in self.entries and self.entries[key][0] is tokenizer:
                self.hits += 1
                self.entries.move_to_end(key)
                return self.entries[key][1]
        value = (tuple(self.tokenize(tokenizer, instruction)),
                 tuple(tuple(self.tokenize(tokenizer, option)) for option in options))
        with self.lock:
            self.misses += 1
            if eligible and len(value[0]) + sum(map(len, value[1])) <= self.max_tokens:
                self.entries[key] = (tokenizer, value)
                self.entries.move_to_end(key)
                while len(self.entries) > self.max_entries:
                    self.entries.popitem(last=False)
        return value
