"""
Scene Analysis Manager - Modularized scene analysis and context management

This module provides a unified interface for:
- Game context extraction (inventory, game state)
- Object detection with consistent logging
- Scene summary generation and display
- Cross-method code reuse and consistency
"""

class SceneAnalyzer:
    """
    High-level scene analysis manager that coordinates between Eye, Detector, and Brain
    for consistent scene analysis across different execution modes.
    """
    
    def __init__(self, eye, detector, brain, gym_adapter):
        """
        Initialize Scene Analyzer with required components.
        
        Args:
            eye: Eye instance for observation and context extraction
            detector: Detector instance for object detection
            brain: Brain instance for scene summary generation
            gym_adapter: Gym environment adapter for game context
        """
        self.eye = eye
        self.detector = detector
        self.brain = brain
        self.gym_adapter = gym_adapter
        self.game_name = eye.game_name if hasattr(eye, 'game_name') else "Unknown"
    
    def analyze_scene_complete(self, observation, step_info="", context_name="SCENE"):
        """
        Complete scene analysis pipeline: context + detection + summary.
        
        Args:
            observation: Game observation dict with screen data
            step_info: Step identifier for logging (e.g., "25", "Step 25")
            context_name: Context name for headers (e.g., "STEP", "MCP", "EXPLORATION")
            
        Returns:
            dict: Complete scene analysis with all components
        """
        # Step 1: Get game context (inventory, game state)
        context = self.eye.get_game_context(self.gym_adapter, observation)
        
        # Step 2: Detect objects with logging
        screen = observation.get('screen')
        if screen is None:
            print(f"⚠️ [{context_name}] No screen data available for analysis")
            return {
                'context': context,
                'detected_objects': [],
                'scene_summary': "No screen data available",
                'analysis_success': False
            }
        
        detected_objects = self.eye.get_detected_objects_with_logging(
            self.detector, screen, f"{context_name} {step_info}".strip()
        )
        
        # Step 3: Generate and display scene summary
        scene_summary = self.eye.generate_and_display_scene_summary(
            self.brain, detected_objects, context['inventory'], context['game_state'],
            step_info=step_info, context_name=context_name
        )
        
        return {
            'context': context,
            'detected_objects': detected_objects,
            'scene_summary': scene_summary,
            'analysis_success': len(detected_objects) > 0,
            'inventory': context['inventory'],
            'game_state': context['game_state']
        }
    
    def analyze_scene_quick(self, observation, context_name="QUICK"):
        """
        Quick scene analysis without detailed logging - for performance-critical paths.
        
        Args:
            observation: Game observation dict with screen data
            context_name: Context name for minimal logging
            
        Returns:
            dict: Basic scene analysis components
        """
        context = self.eye.get_game_context(self.gym_adapter, observation)
        
        detected_objects = []
        screen = observation.get('screen')
        if screen is not None:
            try:
                if hasattr(self.detector, 'get_detected_objects'):
                    detected_objects = self.detector.get_detected_objects(screen)
                elif hasattr(self.detector, 'extract_objects_crafter_api'):
                    detected_objects = self.detector.extract_objects_crafter_api(screen)
                print(f"🔍 [{context_name}] Quick analysis: {len(detected_objects)} objects")
            except Exception as e:
                print(f"❌ [{context_name}] Quick detection failed: {e}")
        
        return {
            'context': context,
            'detected_objects': detected_objects,
            'inventory': context['inventory'],
            'game_state': context['game_state']
        }
    
    def get_context_only(self, observation):
        """
        Get only game context (inventory + game state) without detection.
        Useful for lightweight context queries.
        
        Args:
            observation: Game observation dict
            
        Returns:
            dict: Game context with inventory and game_state
        """
        return self.eye.get_game_context(self.gym_adapter, observation)
    
    def detect_objects_only(self, screen, context_name="DETECTION"):
        """
        Perform only object detection with logging.
        
        Args:
            screen: Screen/image data
            context_name: Context name for logging
            
        Returns:
            list: Detected objects
        """
        return self.eye.get_detected_objects_with_logging(
            self.detector, screen, context_name
        )
    
    def generate_summary_only(self, detected_objects, inventory, game_state, 
                            step_info="", context_name="SUMMARY"):
        """
        Generate only scene summary without detection.
        
        Args:
            detected_objects: Pre-detected objects list
            inventory: Player inventory dict
            game_state: Game state dict
            step_info: Step identifier
            context_name: Context name for header
            
        Returns:
            str: Generated scene summary
        """
        return self.eye.generate_and_display_scene_summary(
            self.brain, detected_objects, inventory, game_state,
            step_info=step_info, context_name=context_name
        )
    
    def validate_components(self):
        """
        Validate that all required components are available and functional.
        
        Returns:
            dict: Validation results for each component
        """
        validation = {
            'eye': self.eye is not None,
            'detector': self.detector is not None,
            'brain': self.brain is not None,
            'gym_adapter': self.gym_adapter is not None,
            'eye_methods': False,
            'detector_methods': False,
            'brain_methods': False
        }
        
        # Check Eye methods
        if self.eye:
            validation['eye_methods'] = (
                hasattr(self.eye, 'get_game_context') and
                hasattr(self.eye, 'get_detected_objects_with_logging') and
                hasattr(self.eye, 'generate_and_display_scene_summary')
            )
        
        # Check Detector methods
        if self.detector:
            validation['detector_methods'] = (
                hasattr(self.detector, 'get_detected_objects') or
                hasattr(self.detector, 'extract_objects_crafter_api')
            )
        
        # Check Brain methods
        if self.brain:
            validation['brain_methods'] = hasattr(self.brain, '_generate_scene_summary')
        
        validation['all_valid'] = all(validation.values())
        
        return validation
