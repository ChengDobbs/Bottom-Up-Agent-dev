import random

class Teacher():
    def __init__(self, config):
        self.type = config['teacher_type']


    def get_operation_guidance(self, candidate_operations):
        if self.type == 'Human':
            return self.get_operation_guidance_human(candidate_operations)
        elif self.type == 'AI':
            return self.get_operation_guidance_ai(candidate_operations)
        elif self.type == 'Random':
            return self.get_operation_guidance_random(candidate_operations)
        else:
            print(f"Unsupported type: {self.type}")
            return None

    def get_operation_guidance_human(self, candidate_operations):
        print("\n[Human Operation Guidance]")
        print("Please select an operation by typing the index:")

        for idx, op in enumerate(candidate_operations):
            print(f"{idx}: {op}")

        while True:
            try:
                choice = input("Enter index of chosen operation: ").strip()
                if choice == '':
                    print("No input received. Returning None.")
                    return None
                if not choice.isdigit():
                    print("Invalid input. Please enter a valid index.")
                    continue

                index = int(choice)
                if 0 <= index < len(candidate_operations):
                    return candidate_operations[index]
                else:
                    print("Index out of range. Try again.")
            except KeyboardInterrupt:
                print("\nCancelled by user.")
                return None
            except Exception as e:
                print(f"Error: {e}")
                return None

    def get_operation_guidance_ai(self, candidate_operations):
        print("\n[AI Operation Guidance]")
        print(f"AI will select an operation randomly: {candidate_operations}")
        return random.choice(candidate_operations)
    
    def get_operation_guidance_random(self, candidate_operations):
        return random.choice(candidate_operations)
