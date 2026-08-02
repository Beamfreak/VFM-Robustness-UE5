// Copyright (c) 2025 Florian Gutbier
// 
// This source code is part of the UE5 Plugin developed for the Bachelor's thesis
// at the University of Bamberg.
// 
// Released under the MIT License. See LICENSE file for details.
#pragma once

#include "Kismet/BlueprintFunctionLibrary.h"
#include "Engine/World.h"
#include "Engine/EngineTypes.h"
#include "Engine/StaticMesh.h"
#include "GameFramework/Actor.h"
#include "GameFramework/PlayerController.h"
#include "Engine/Engine.h"
#include "Misc/Paths.h"
#include "Engine/TargetPoint.h"
#include "HAL/PlatformFileManager.h"
#include "TimerManager.h"
#include "DatasetRendererBachelorThesisBPLibrary.generated.h"

/**
 * @brief Blueprint function library to trigger dataset rendering from Blueprints.
 *
 * Provides entry points for starting the dataset capture process
 * by instantiating and controlling the DatasetCaptureManager.
 */
UCLASS()
class UBachelorRenderingBPLibrary : public UBlueprintFunctionLibrary
{
    GENERATED_BODY()

public:

    /**
     * @brief Starts the dataset capture process using a CSV file for mesh-class mapping.
     *
     * @param WorldContextObject The world context.
     * @param ObjectTarget A scene reference point for positioning the model.
     * @param CameraTargets An array of camera direction vectors.
     * @param LightColors An array of light colors to iterate over.
     * @param Materials An array of materials to iterate over.
     * @param MeshClassCSVPath Path to a CSV file mapping mesh asset paths to class IDs.
     * @param addFog Whether to add a local fog volume.
     * @param TargetRadius Target bounding sphere radius for normalizing mesh sizes (cm).
     * @param NextLevel Optional. If provided, this level will be loaded after completion.
     * @param MaskMaterial Optional unlit material applied to the object for segmentation mask rendering.
     * @param bGenerateMasks Whether to generate segmentation masks alongside each screenshot.
     */
    UFUNCTION(BlueprintCallable, Category = "Dataset", meta = (WorldContext = "WorldContextObject"))
    static void StartDatasetCapture(
        UObject* WorldContextObject,
        ATargetPoint* ObjectTarget,
        const TArray<FVector>& CameraTargets,
        const TArray<FLinearColor>& LightColors,
        const TArray<UMaterialInterface*>& Materials,
        const FString& MeshClassCSVPath,
        bool addFog,
        float TargetRadius = 50.f,
        TSoftObjectPtr<UWorld> NextLevel = nullptr,
        UMaterialInterface* MaskMaterial = nullptr,
        bool bGenerateMasks = false
    );
};
