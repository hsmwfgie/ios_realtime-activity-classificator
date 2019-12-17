//
//  ActivityClassifier.swift
//  PoseEstimation-CoreML
//
//  Created by Tony Rolletschke on 04.12.19.
//  Copyright © 2019 tucan9389. All rights reserved.
//

import UIKit
import Vision

protocol ActivityManagerDelegate {
    
    func didUpdateActivity(prediction :(activity: String?,probability:Double?))
    
}

class ActivityManager {

    private let activityClassificationModel = MyActivityClassifierTony()
    
    struct ModelConstants {
        static let numOfFeatures = 16
        static let predictionWindowSize = 15
        static let hiddenInLength = 200
        static let hiddenCellInLength = 200
    }
    
    private var predictionWindowDataArray : MLMultiArray? = try? MLMultiArray(shape: [1 , ModelConstants.predictionWindowSize , ModelConstants.numOfFeatures] as [NSNumber], dataType: MLMultiArrayDataType.double)
    private var lastHiddenCellOutput: MLMultiArray?
    private var lastHiddenOutput: MLMultiArray?
    private var currentIndexInPredictionWindow = 0
        
    var delegate: ActivityManagerDelegate?
    
    var videoWidth : CGFloat = 1920.00
    var videoHeigth : CGFloat = 1440.00
    
    init() {
        //
        self.setUpActivtyClassifier()
    }
    
    deinit {
        self.predictionWindowDataArray = nil
        self.lastHiddenCellOutput = nil
        self.lastHiddenOutput = nil
        self.currentIndexInPredictionWindow = 0
    }
    
    func setUpActivtyClassifier(){
        
        predictionWindowDataArray = try? MLMultiArray(shape: [1 , ModelConstants.predictionWindowSize , ModelConstants.numOfFeatures] as [NSNumber], dataType: MLMultiArrayDataType.double)
        lastHiddenOutput = try? MLMultiArray(shape:[ModelConstants.hiddenInLength as NSNumber], dataType: MLMultiArrayDataType.double)
        lastHiddenCellOutput = try? MLMultiArray(shape:[ModelConstants.hiddenCellInLength as NSNumber], dataType: MLMultiArrayDataType.double)
        
    }
    
    func predictActivty(bodypoints :Any) {

        //guard let upperBody = getUpperBodyFromPoseEstimation(points: bodypoints as! [BodyPoint?]) else {return}
        let upperBody:UpperBody? = getUpperBodyFromARKit(points: bodypoints as! [Int:CGPoint])
        if upperBody != nil {
            predictUpperBodyActivity(upperBody: upperBody!)
        }else{
            delegate?.didUpdateActivity(prediction: ("unknown",0.0))
        }
    }
    
    func predictUpperBodyActivity(upperBody : UpperBody) {
        
         guard let dataArray = predictionWindowDataArray else {
            delegate?.didUpdateActivity(prediction: ("unknown",0.0))
            return
        }

        if upperBody.rshoulder.point != nil {
            dataArray[[0 , currentIndexInPredictionWindow ,0] as [NSNumber]] = Double(upperBody.rshoulder.point!.x) as NSNumber
            dataArray[[0 , currentIndexInPredictionWindow ,1] as [NSNumber]] = Double(upperBody.rshoulder.point!.y) as NSNumber
        }else{
            return
        }
        if upperBody.relbow.point != nil {
            dataArray[[0 , currentIndexInPredictionWindow ,2] as [NSNumber]] = Double(upperBody.relbow.point!.x) as NSNumber
            dataArray[[0 , currentIndexInPredictionWindow ,3] as [NSNumber]] = Double(upperBody.relbow.point!.y) as NSNumber
        }else{
            return
        }
        if upperBody.rwrist.point != nil {
            dataArray[[0 , currentIndexInPredictionWindow ,4] as [NSNumber]] = Double(upperBody.rwrist.point!.x) as NSNumber
            dataArray[[0 , currentIndexInPredictionWindow ,5] as [NSNumber]] = Double(upperBody.rwrist.point!.y) as NSNumber
        }else{
            return
        }
        if upperBody.lshoulder.point != nil {
            dataArray[[0 , currentIndexInPredictionWindow ,6] as [NSNumber]] = Double(upperBody.lshoulder.point!.x) as NSNumber
            dataArray[[0 , currentIndexInPredictionWindow ,7] as [NSNumber]] = Double(upperBody.lshoulder.point!.y) as NSNumber
        }else{
            return
        }
        if upperBody.lelbow.point != nil {
            dataArray[[0 , currentIndexInPredictionWindow ,8] as [NSNumber]] = Double(upperBody.lelbow.point!.x) as NSNumber
            dataArray[[0 , currentIndexInPredictionWindow ,9] as [NSNumber]] = Double(upperBody.lelbow.point!.y) as NSNumber
        }else{
            return
        }
        if upperBody.lwrist.point != nil {
            dataArray[[0 , currentIndexInPredictionWindow ,10] as [NSNumber]] = Double(upperBody.lwrist.point!.x) as NSNumber
            dataArray[[0 , currentIndexInPredictionWindow ,11] as [NSNumber]] = Double(upperBody.lwrist.point!.y) as NSNumber
        }else{
            return
        }
        if upperBody.rhip.point != nil {
            dataArray[[0 , currentIndexInPredictionWindow ,12] as [NSNumber]] = Double(upperBody.rhip.point!.x) as NSNumber
            dataArray[[0 , currentIndexInPredictionWindow ,13] as [NSNumber]] = Double(upperBody.rhip.point!.y) as NSNumber
        }else{
            return
        }
        if upperBody.lhip.point != nil {
            dataArray[[0 , currentIndexInPredictionWindow ,14] as [NSNumber]] = Double(upperBody.lhip.point!.x) as NSNumber
            dataArray[[0 , currentIndexInPredictionWindow ,15] as [NSNumber]] = Double(upperBody.lhip.point!.y) as NSNumber
        }else{
            return
        }

         // Update the index in the prediction window data array
         currentIndexInPredictionWindow += 1

         // If the data array is full, call the prediction method to get a new model prediction.
         // We assume here for simplicity that the Gyro data was added to the data array as well.
         if (currentIndexInPredictionWindow == ModelConstants.predictionWindowSize) {
             let predictedActivity = performModelPrediction()

            //print("Activity: ", predictedActivity.activity!)
            
            if(predictedActivity.probability?.isNaN == true) {
                self.predictionWindowDataArray = nil
                self.lastHiddenCellOutput = nil
                self.lastHiddenOutput = nil
                self.setUpActivtyClassifier()
                currentIndexInPredictionWindow = 0
                delegate?.didUpdateActivity(prediction: ("Initializing...",0.0))
                return
            }
            
            delegate?.didUpdateActivity(prediction: predictedActivity)
             // Start a new prediction window
             currentIndexInPredictionWindow = 0
         }
        
    }
    
    
    func normalizeBodyPoint(bodyPoint:CGPoint, topBodyPoint: CGPoint, normFactor:Double) -> CGPoint {
        
        let normX = (Double(bodyPoint.x * videoWidth)-Double(topBodyPoint.x * videoWidth))/normFactor
        let normY = (Double(bodyPoint.y * videoHeigth)-Double(topBodyPoint.y * videoHeigth))/normFactor
        
        return CGPoint(x: normX, y: normY)
        
    }
    
    private func getUpperBodyFromARKit(points: [Int:CGPoint]) -> UpperBody? {

        var upperBody = UpperBody()
        
        if(points[1] != nil) {
            upperBody.topBodyPart.point = points[1]
        }
        
        if(points[16] != nil) {
            upperBody.buttomBodyPart.point = points[16]
        }
        else{
            if(points[8] != nil) {
                upperBody.buttomBodyPart.point = points[8]
            }else{
                if(points[11] != nil) {
                    upperBody.buttomBodyPart.point = points[11]
                }else{
                    return nil
                }
            }
        }
        
        if upperBody.topBodyPart.point != nil && upperBody.buttomBodyPart.point != nil {

            let normFactor : Double = Double(upperBody.buttomBodyPart.point!.y * videoHeigth) - Double(upperBody.topBodyPart.point!.y * videoHeigth)


            if let rShoulder : CGPoint = points[2]{
                upperBody.rshoulder.point = normalizeBodyPoint(bodyPoint: rShoulder, topBodyPoint: upperBody.topBodyPart.point!, normFactor: normFactor)
            }else{return nil}
            if let rElbow: CGPoint = points[3] {
                upperBody.relbow.point = normalizeBodyPoint(bodyPoint: rElbow, topBodyPoint: upperBody.topBodyPart.point!, normFactor: normFactor)
            }else{return nil}
            if let rWrist: CGPoint = points[4] {
                upperBody.rwrist.point = normalizeBodyPoint(bodyPoint: rWrist, topBodyPoint: upperBody.topBodyPart.point!, normFactor: normFactor)
            }else{return nil}
            if let lShoulder : CGPoint = points[5] {
                upperBody.lshoulder.point = normalizeBodyPoint(bodyPoint: lShoulder, topBodyPoint: upperBody.topBodyPart.point!, normFactor: normFactor)
            }else{return nil}
            if let lElbow: CGPoint = points[6] {
                upperBody.lelbow.point = normalizeBodyPoint(bodyPoint: lElbow, topBodyPoint: upperBody.topBodyPart.point!, normFactor: normFactor)
            }else{return nil}
            if let lWrist: CGPoint = points[7] {
                upperBody.lwrist.point = normalizeBodyPoint(bodyPoint: lWrist, topBodyPoint: upperBody.topBodyPart.point!, normFactor: normFactor)
            }else{return nil}
            if let rHip: CGPoint = points[8] {
                upperBody.rhip.point = normalizeBodyPoint(bodyPoint: rHip, topBodyPoint: upperBody.topBodyPart.point!, normFactor: normFactor)
            }else{return nil}
            if let lHip: CGPoint = points[11] {
                    upperBody.lhip.point = normalizeBodyPoint(bodyPoint: lHip, topBodyPoint: upperBody.topBodyPart.point!, normFactor: normFactor)
            }else{return nil}
        }
        
        //print(upperBody)
        return upperBody
    }

//    private func getUpperBodyFromPoseEstimation(points: [BodyPoint?]) -> UpperBody? {
//
//        var upperBody = UpperBody()
//
//        // top body point is equal to neck
//        guard let neck : BodyPoint = points[1] else {return nil}
//        upperBody.topBodyPart.point = neck.maxPoint
//
//        // botton point is equal to hip right or left
//        if let hipRight : BodyPoint = points[8] {
//            upperBody.buttomBodyPart.point = hipRight.maxPoint
//        }
//        else if let hipLeft : BodyPoint = points[11]{
//            upperBody.buttomBodyPart.point = hipLeft.maxPoint
//        }
//        else{
//            return nil
//        }
//
//        //
//        if upperBody.topBodyPart.point != nil && upperBody.buttomBodyPart.point != nil {
//
//            let normFactor : Double = Double(upperBody.buttomBodyPart.point!.y * videoHeigth) - Double(upperBody.topBodyPart.point!.y * videoHeigth)
//
//            if let rShoulder : CGPoint = points[2]?.maxPoint {
//                upperBody.rshoulder.point = normalizeBodyPoint(bodyPoint: rShoulder, topBodyPoint: upperBody.topBodyPart.point!, normFactor: normFactor)
//            }else{return nil}
//            if let rElbow: CGPoint = points[3]?.maxPoint {
//                upperBody.relbow.point = normalizeBodyPoint(bodyPoint: rElbow, topBodyPoint: upperBody.topBodyPart.point!, normFactor: normFactor)
//            }else{return nil}
//            if let rWrist: CGPoint = points[4]?.maxPoint {
//                upperBody.rwrist.point = normalizeBodyPoint(bodyPoint: rWrist, topBodyPoint: upperBody.topBodyPart.point!, normFactor: normFactor)
//            }else{return nil}
//            if let lShoulder : CGPoint = points[5]?.maxPoint {
//                upperBody.lshoulder.point = normalizeBodyPoint(bodyPoint: lShoulder, topBodyPoint: upperBody.topBodyPart.point!, normFactor: normFactor)
//            }else{return nil}
//            if let lElbow: CGPoint = points[6]?.maxPoint {
//                upperBody.lelbow.point = normalizeBodyPoint(bodyPoint: lElbow, topBodyPoint: upperBody.topBodyPart.point!, normFactor: normFactor)
//            }else{return nil}
//            if let lWrist: CGPoint = points[7]?.maxPoint {
//                upperBody.lwrist.point = normalizeBodyPoint(bodyPoint: lWrist, topBodyPoint: upperBody.topBodyPart.point!, normFactor: normFactor)
//            }else{return nil}
//            if let rHip: CGPoint = points[8]?.maxPoint {
//                upperBody.rhip.point = normalizeBodyPoint(bodyPoint: rHip, topBodyPoint: upperBody.topBodyPart.point!, normFactor: normFactor)
//            }else{return nil}
//            if let lHip: CGPoint = points[11]?.maxPoint {
//                upperBody.lhip.point = normalizeBodyPoint(bodyPoint: lHip, topBodyPoint: upperBody.topBodyPart.point!, normFactor: normFactor)
//            }else{return nil}
//        }
//
//        return upperBody
//    }
    
    func performModelPrediction () -> (activity: String?,probability:Double?) {
        guard let dataArray = predictionWindowDataArray else { return (activity: "Error!",probability: 0)}
        
        var probability:Double? = 0
        // Perform model prediction
        //print("dataArray: \(dataArray[7])")
        let modelPrediction = try? activityClassificationModel.prediction(features: dataArray, hiddenIn: lastHiddenOutput, cellIn: lastHiddenCellOutput)
        
        // Update the state vectors
        lastHiddenOutput = modelPrediction?.hiddenOut
        lastHiddenCellOutput = modelPrediction?.cellOut
        
        if let prediction = modelPrediction {
            probability = prediction.activityProbability[prediction.activity]
        }
        
        // Return the predicted activity - the activity with the highest probability
        return (activity: modelPrediction?.activity,probability: probability)
    }
    
    struct UpperBody {
        
        var rshoulder = UpperBodyPoint()
        var lshoulder = UpperBodyPoint()
        var relbow = UpperBodyPoint()
        var lelbow = UpperBodyPoint()
        var rwrist = UpperBodyPoint()
        var lwrist = UpperBodyPoint()
        var rhip = UpperBodyPoint()
        var lhip = UpperBodyPoint()
        var rankle = UpperBodyPoint()
        var lankle = UpperBodyPoint()
        
        var topBodyPart = UpperBodyPoint()
        var buttomBodyPart = UpperBodyPoint()
        
    }

    struct UpperBodyPoint {
        //var point : CGPoint = CGPoint(x: 0.0, y: 0.0)
        var point : CGPoint?
        var score : Double?
    }
    
}
