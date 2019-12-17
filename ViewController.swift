/*
See LICENSE folder for this sample’s licensing information.

Abstract:
The sample app's main view controller.
*/

import UIKit
import RealityKit
import ARKit
import Combine

class ViewController: UIViewController, ARSessionDelegate, ActivityManagerDelegate {
    
    func didUpdateActivity(prediction: (activity: String?,probability:Double?)) {
        DispatchQueue.main.async {
           self.messageLabel.text = "Activity: \(prediction.activity ?? "unknown") (\(prediction.probability ?? 0.0))"
        }
        
    }
    

    @IBOutlet weak var messageLabel: UILabel!
    @IBOutlet var arView: ARView!
    @IBOutlet weak var keypointView: UIView!
    
    var keypoints = [Int:UIView]()
    var imageSizeKeypoints = [Int:CGPoint]()
    //var normedKeypoints = [Int:simd_float2]()
    var waitCounter:Int = 0
    
    // The 3D character to display.
    //var character: BodyTrackedEntity?
    //let characterOffset: SIMD3<Float> = [-1.0, 0, 0] // Offset the character by one meter to the left
    //let characterAnchor = AnchorEntity()
    let keypointColors = [UIColor.red,UIColor.green,UIColor.gray,UIColor.darkGray,UIColor.lightGray,UIColor.purple,UIColor.systemPink,UIColor.blue,UIColor.black,UIColor.yellow,UIColor.cyan,UIColor.systemOrange,UIColor.white,UIColor.brown,UIColor.magenta,UIColor.orange,UIColor.green]
    
    // A tracked raycast which is used to place the character accurately
    // in the scene wherever the user taps.
    //var placementRaycast: ARTrackedRaycast?
    //var tapPlacementAnchor: AnchorEntity?
    
    var activityManager:ActivityManager? = nil
    
    override func viewDidAppear(_ animated: Bool) {
        super.viewDidAppear(animated)
        arView.session.delegate = self
        
        guard ARBodyTrackingConfiguration.isSupported else {
            fatalError("This feature is only supported on devices with an A12 chip")
        }

        // Run a body tracking configration.
        let configuration = ARBodyTrackingConfiguration()
        configuration.frameSemantics = .bodyDetection
        arView.session.run(configuration)

    }
    
    override func viewDidDisappear(_ animated: Bool) {
        activityManager = nil
    }
        
    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        let ImageWidth = self.arView.frame.size.width
        let ImageHeight = (frame.camera.imageResolution.height * self.arView.frame.width) / frame.camera.imageResolution.width
        
        
        let diffHeight:CGFloat = 115.0
        
        if waitCounter == 0 {
            waitCounter += 1
            print("Camera: \(frame.camera.imageResolution.width) x \(frame.camera.imageResolution.height)")
            print("Screen: \(self.arView.frame.width) x \(self.arView.frame.height)")
            print("Verwendet: \(ImageWidth) x \(ImageHeight)")
            
            activityManager = ActivityManager.init()
            activityManager!.videoWidth = self.arView.frame.width
            activityManager!.videoHeigth = self.arView.frame.height
            activityManager!.delegate = self

            return
        }
        
        let person = frame.detectedBody
        let skeleton2D = person?.skeleton
        
        if let jointLandmarks = skeleton2D?.jointLandmarks {
            for (i,joint) in jointLandmarks.enumerated() {
                if(joint.x.isNaN == false && joint.y.isNaN == false){
                    self.imageSizeKeypoints[i] = CGPoint.init(x: CGFloat(joint.x), y: CGFloat(joint.y))
                    
                    //Visualize the body points
                    var myView = arView.viewWithTag(i+10)
                    if(myView == nil){
                        myView = UIView.init(frame: CGRect.init(x: (CGFloat(joint.x) * ImageWidth)-5.0, y: (CGFloat(joint.y) + ImageHeight)-5.0-diffHeight, width: 10.0, height: 10.0))
                        myView?.tag = i+10
                        keypoints[i] = myView
                    }else{
                        keypoints[i]?.frame.origin.x = (CGFloat(joint.x) * ImageWidth)-5.0
                        keypoints[i]?.frame.origin.y = (CGFloat(joint.y) * ImageHeight)-5.0-diffHeight
                    }
                    keypoints[i]?.layer.cornerRadius = 10
                    keypoints[i]?.backgroundColor = keypointColors[i]
                    self.arView.addSubview(keypoints[i]!)
                    
                }else{
                    self.imageSizeKeypoints[i] = nil
                }
//                head_joint [0]
//                neck_1_joint [1]
//                right_shoulder_1_joint [2]
//                right_forearm_joint [3]
//                right_hand_joint [4]
//                left_shoulder_1_joint [5]
//                left_forearm_joint [6]
//                left_hand_joint [7]
//                right_upLeg_joint [8]
//                right_leg_joint [9]
//                right_foot_joint [10]
//                left_upLeg_joint [11]
//                left_leg_joint [12]
//                left_foot_joint [13]
//                right_eye_joint [14]
//                left_eye_joint [15]
//                root [16]
            }
        }
        
        // Detect Activity
        activityManager!.predictActivty(bodypoints: self.imageSizeKeypoints)
    }
}
