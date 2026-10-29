"""Presentation-only streaming formatter; never dispatches or executes a tool."""

import re


class Transcript:
    TAGS = {'<tool_call>': '</tool_call>', '<tool_response>': '</tool_response>',
            '<think>': '</think>'}

    def __init__(self, write):
        self.write, self.buffer, self.tag = write, '', None

    def feed(self, text):
        self.buffer += text
        while self.buffer:
            if self.tag:
                end = self.TAGS[self.tag]
                at = self.buffer.find(end)
                if at < 0:
                    if len(self.buffer) > 16384:
                        raise ValueError('Model markup exceeded the display buffer limit')
                    return
                body, self.buffer = self.buffer[:at], self.buffer[at + len(end):]
                if self.tag == '<tool_call>':
                    name = re.search(r'<function=([\w.-]+)>', body)
                    match = re.search(r'<parameter=[\w.-]+>\s*(.*?)\s*</parameter>', body, re.S)
                    command = match[1].replace('\n', ' ')[:120] if match else 'requested'
                    self.write('\n┌ ' + (name[1] if name else 'harness') + ' · ' + command + '\n')
                elif self.tag == '<tool_response>':
                    for line in body.strip().splitlines():
                        self.write('│ ' + line + '\n')
                    self.write('└ neural guest\n')
                self.tag = None
            else:
                positions = [(self.buffer.find(tag), tag) for tag in self.TAGS if tag in self.buffer]
                if positions:
                    at, tag = min(positions)
                    self.write(self.buffer[:at])
                    self.buffer, self.tag = self.buffer[at + len(tag):], tag
                    continue
                keep = max((length for tag in self.TAGS for length in range(1, len(tag))
                            if self.buffer.endswith(tag[:length])), default=0)
                if keep:
                    self.write(self.buffer[:-keep])
                    self.buffer = self.buffer[-keep:]
                else:
                    self.write(self.buffer)
                    self.buffer = ''
                return

    def finish(self):
        if not self.tag and self.buffer:
            self.write(self.buffer)
        elif self.tag:
            self.write('\n[incomplete model block]\n')
        self.buffer, self.tag = '', None
