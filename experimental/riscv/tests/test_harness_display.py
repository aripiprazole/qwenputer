from chipformer.rv.harness_display import Transcript


def test_streamed_tool_markup_is_only_formatted_not_executed():
    output = []
    display = Transcript(output.append)
    text = ('Checking. <tool_call><function=harness><parameter=command>'
            '9 square .</parameter></function></tool_call>'
            '<tool_response>81</tool_response><think>private reasoning</think>Done.')
    for char in text:
        display.feed(char)
    display.finish()
    rendered = ''.join(output)
    assert 'Checking.' in rendered and 'Done.' in rendered
    assert 'harness · 9 square .' in rendered
    assert '│ 81' in rendered and 'neural guest' in rendered
    assert '<tool_call>' not in rendered and 'private reasoning' not in rendered


def test_unfinished_call_does_not_invent_a_tool_result():
    output = []
    display = Transcript(output.append)
    display.feed('<tool_call><function=harness>')
    display.finish()
    assert ''.join(output) == '\n[incomplete model block]\n'
