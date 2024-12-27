import tensorflow as tf

file_path = 'D:\Art_detector\logs\events.out.tfevents.1731324751.DESKTOP-PAV2DVN.23000.0'

for event in tf.compat.v1.train.summary_iterator(file_path):
    for value in event.summary.value:
        print(f"Step: {event.step}, Tag: {value.tag}, Value: {value.simple_value if value.HasField('simple_value') else 'N/A'}")
