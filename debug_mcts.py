"""Debug MCTS service issues."""

import json
import subprocess
import sys
import time

# Test 1: Can we start the service?
print("Test 1: Starting MCTS service...")
try:
    proc = subprocess.Popen(
        ['node', 'mcts_service.js'],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    time.sleep(0.5)  # Give it time to crash if it's going to
    if proc.poll() is not None:
        stderr = proc.stderr.read()
        print(f"ERROR: Service crashed immediately")
        print(f"Exit code: {proc.returncode}")
        print(f"STDERR: {stderr}")
        sys.exit(1)
    print("OK: Service started")
except Exception as e:
    print(f"ERROR: Failed to start service: {e}")
    sys.exit(1)

# Test 2: Send minimal request
print("\nTest 2: Sending minimal request...")
minimal_request = {
    "state": {
        "own_team": [{
            "species": "Pikachu",
            "level": 50,
            "moves": ["thunderbolt", "quickattack", "irontail", "thunderwave"],
            "ability": "static",
            "item": "lightball",
            "hp": 95,
            "maxhp": 95,
            "status": None,
            "boosts": {},
            "active": True,
            "fainted": False,
            "gender": "M",
        }],
        "opp_team": [{
            "species": "Charizard",
            "level": 50,
            "moves": ["flamethrower"],
            "ability": None,
            "item": None,
            "hp_fraction": 1.0,
            "status": None,
            "boosts": {},
            "active": True,
            "fainted": False,
            "revealed": True,
            "gender": "M",
        }],
        "opp_team_size": 6,
        "field": {"weather": None, "terrain": None, "trick_room": False},
        "side_conditions": {},
        "opp_side_conditions": {},
        "turn": 1,
    },
    "priors": [1/26] * 26,
    "value": 0.0,
    "action_mask": [0]*6 + [1]*4 + [0]*16,
}

try:
    # Send request
    request_str = json.dumps(minimal_request) + '\n'
    print(f"Request size: {len(request_str)} bytes")
    proc.stdin.write(request_str)
    proc.stdin.flush()
    print("OK: Request sent")

    # Wait for response
    print("\nWaiting for response (timeout 10s)...")

    # Try to read with timeout
    import select
    import threading

    result = {'response': None, 'stderr': None}

    def read_response():
        try:
            result['response'] = proc.stdout.readline()
        except Exception as e:
            result['error'] = str(e)

    thread = threading.Thread(target=read_response)
    thread.daemon = True
    thread.start()
    thread.join(timeout=10)

    if result['response']:
        print(f"OK: Got response ({len(result['response'])} bytes)")
        print(f"Response preview: {result['response'][:200]}")
        try:
            response = json.loads(result['response'])
            print(f"Best action: {response.get('bestAction')}")
            print(f"Visits sum: {sum(response.get('visits', []))}")
        except Exception as e:
            print(f"ERROR: Failed to parse response: {e}")
    else:
        print("ERROR: No response received")

        # Check stderr
        try:
            # Non-blocking read attempt
            proc.stderr.flush()
            stderr_data = proc.stderr.read(1000)  # Read up to 1000 chars
            if stderr_data:
                print(f"\nSTDERR output:")
                print(stderr_data)
        except:
            pass

        # Check if process crashed
        poll_result = proc.poll()
        if poll_result is not None:
            print(f"\nERROR: Process exited with code {poll_result}")
        else:
            print("\nWARNING: Process still running but no output")

except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()

finally:
    proc.terminate()
    try:
        proc.wait(timeout=2)
    except:
        proc.kill()
    print("\nService terminated")
