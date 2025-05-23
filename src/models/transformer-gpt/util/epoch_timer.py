"""
@author : Gabryel-lima
@when : 2025-01-30
@homepage : https://github.com/Gabryel-lima
"""

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs
