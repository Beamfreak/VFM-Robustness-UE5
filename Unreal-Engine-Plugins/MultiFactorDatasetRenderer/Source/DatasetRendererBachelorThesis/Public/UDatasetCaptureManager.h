// Copyright (c) 2025 Florian Gutbier
// 
// This source code is part of the UE5 Plugin developed for the Bachelor's thesis
// at the University of Bamberg.
// 
// Released under the MIT License. See LICENSE file for details.

#pragma once

#include "CoreMinimal.h"
#include "Engine/World.h"
#include "Engine/EngineTypes.h"
#include "Engine/StaticMesh.h"
#include "GameFramework/Actor.h"
#include "GameFramework/PlayerController.h"
#include "Engine/Engine.h"
#include "Engine/TargetPoint.h"
#include "Engine/LocalFogVolume.h"
#include "Misc/Paths.h"
#include "HAL/PlatformFileManager.h"
#include "TimerManager.h"
#include "DatasetMetadataWriter.h"
class USceneCaptureComponent2D;
class UTextureRenderTarget2D;

#include "UDatasetCaptureManager.generated.h"

/**
 * @brief Enum to track different capture phases.
 * Captures vary ONE factor at a time to create a more controlled dataset.
 */
UENUM(BlueprintType)
enum class ECapturePhase : uint8
{
	CameraVariations = 0,     // Vary camera positions (default light, material, no fog)
	LightVariations = 1,      // Vary light colors (reference camera)
	MaterialVariations = 2,   // Vary materials (reference camera, default light)
	FogVariation = 3,         // Capture with fog (reference camera, default settings)
	BackgroundVariations = 4, // Load next background and capture all meshes with default settings
	Complete = 5              // All phases complete for current mesh/background
};

UCLASS()
class DATASETRENDERERBACHELORTHESIS_API UDatasetCaptureManager : public UObject
{
	GENERATED_BODY()

public:

	// Static variables to persist state across level loads
	static FString s_DefaultBackgroundName;
	static bool s_bIsFirstCaptureEver;
	static bool s_bDummyScreenshotDone;

	/**
	 * @brief Initializes the capture manager with scene, mesh, and camera configuration.
	 *
	 * @param InWorld The world context in which meshes are spawned.
	 * @param InObjectTarget A reference point used for mesh placement and camera orientation.
	 * @param InCameraTargets A list of directional vectors indicating camera positions.
	 * @param LightColors A list of light colors to apply during capture.
	 * @param Materials A list of materials to apply during capture.
	 * @param InMeshClassEntries Mesh-to-class-ID pairs (loaded from CSV or provided directly).
	 * @param NextLevel The level to load after capture is complete.
	 * @param addFog Whether to add a local fog volume.
	 * @param InTargetRadius Target bounding sphere radius for normalizing mesh sizes (in cm).
	 * @param InMaskMaterial Unlit material applied to the object for segmentation mask rendering (optional).
	 */
	void Initialize(
		UWorld* InWorld,
		ATargetPoint* InObjectTarget,
		const TArray<FVector>& InCameraTargets,
		const TArray<FLinearColor>& LightColors,
		const TArray<UMaterialInterface*>& Materials,
		const TArray<TPair<TSoftObjectPtr<UStaticMesh>, int32>>& InMeshClassEntries,
		const TSoftObjectPtr<UWorld>& NextLevel,
		bool addFog,
		float InTargetRadius = 50.f,
		UMaterialInterface* InMaskMaterial = nullptr
	);

	/**
	 * @brief Loads a mesh-to-class mapping from a CSV file.
	 *
	 * Expected CSV format (semicolon-delimited, one entry per line):
	 *   /Game/Path/To/Mesh.MeshName;ClassIndex
	 *
	 * @param CSVFilePath Absolute or project-relative path to the CSV file.
	 * @param OutEntries The resulting mesh-class pairs.
	 * @return True if the file was loaded and at least one valid entry was parsed.
	 */
	static bool LoadMeshClassMapFromCSV(
		const FString& CSVFilePath,
		TArray<TPair<TSoftObjectPtr<UStaticMesh>, int32>>& OutEntries
	);

	void StartCapture();

private:

	// Store pointers as UPROPERTY to ensure proper functionality of garbage collector.

	UPROPERTY()
	UWorld* m_pWorld;

	UPROPERTY()
	ATargetPoint* m_pObjectTarget;

	UPROPERTY()
	TArray<FVector> m_aCameraTargets;

	UPROPERTY()
	TArray<UMaterialInterface*> m_aMaterials;

	UPROPERTY()
	TSoftObjectPtr<UWorld> m_pNextLevel;

	UPROPERTY()
	AActor* m_pCurrentSpawnedActor = nullptr;

	UPROPERTY()
	UDatasetMetadataWriter* m_pMetadataWriter;

	UPROPERTY()
	ALocalFogVolume* m_pLocalFog;

	TArray<TPair<TSoftObjectPtr<UStaticMesh>, int32>> m_aMeshEntries;

	float m_fTargetRadius;

	TArray<FLinearColor> m_aLightColors;

	UPROPERTY()
	FVector m_vCurrentObjectCenter;

	float   m_fCurrentObjectRadius;

	int32 m_iCurrentMeshIndex;

	int32 m_iCurrentCameraIndex;

	int32 m_iCurrentLightColorIndex;

	UPROPERTY()
	TArray<FVector> m_vCameraTransforms;

	int32 m_iCurrentMaterialIndex;

	bool m_bFogActive;

	bool m_bAddFog;

	FDelegateHandle m_oScreenshotCapturedHandle;

	FString m_sCurrentScreenshotPath;

	FString m_sRelativeImagePath;

	// ---- Original map lighting (captured at initialization) ----
	// Maps light actor to its original color for restoration
	TMap<AActor*, FLinearColor> m_OriginalLightColors;

	// ---- Segmentation mask members ----

	UPROPERTY()
	UMaterialInterface* m_pMaskMaterial = nullptr;

	UPROPERTY()
	AActor* m_pCaptureActor = nullptr;

	UPROPERTY()
	USceneCaptureComponent2D* m_pMaskCaptureComponent = nullptr;

	UPROPERTY()
	UTextureRenderTarget2D* m_pMaskRenderTarget = nullptr;

	FString m_sCurrentMaskPath;

	FString m_sRelativeMaskPath;

	// ---- Phase-based capture flow ----

	ECapturePhase m_eCapturePhase = ECapturePhase::CameraVariations;

	// Store the default background name to detect background variations
	FString m_sDefaultBackgroundName;

	// Store original materials to restore them between material variations
	TMap<UMeshComponent*, TArray<UMaterialInterface*>> m_OriginalMaterials;

	// Track shader compilation wait iterations during level streaming
	int32 m_iWaitIterationCount = 0;

	FString GetDatasetRoot() const;

	void SetupMeshEntries(const TArray<TPair<TSoftObjectPtr<UStaticMesh>, int32>>& InMeshClassEntries);

	void CreateScreenshotFolder() const;

	void SetUpFog();

	void BuildCameraTargetsForCurrentActor();

	void ProcessCaptureState();

	void SpawnCurrentMesh();

	/** Polls until all streaming textures on the current actor are fully loaded. */
	void WaitForTextureStreaming();

	void NormalizeActorScale();

	void CaptureScreenshotForCurrentCamera();

	void RequestCameraScreenshot();

	/** Polls until the level's streaming textures are fully loaded before starting capture. */
	void WaitForLevelStreaming();

	ALocalFogVolume* CreateTempFogVolume(const FVector& Location, const FVector& UniformScale, float RadialDensity, float HeightDensity, const FLinearColor& Albedo, float PhaseG);

	void SetAllLightsColor(const FLinearColor& NewColor) const;

	/** Capture the current light colors from all lights in the world. */
	void CaptureOriginalLightColors();

	/** Restore all lights to their original colors captured at initialization. */
	void RestoreOriginalLightColors() const;

	void ApplyCurrentMaterial();

	void RestoreOriginalMaterials();

	void OnScreenshotCaptured(int32 Width, int32 Height, const TArray<FColor>& Bitmap);

	/** Sets up the SceneCaptureComponent2D and render target for segmentation masks. */
	void SetupMaskCapture();

	/** Captures and saves a segmentation mask for the current camera position. */
	void CaptureSegmentationMask(int32 ScreenshotWidth, int32 ScreenshotHeight);

	void FinalizeCapture();

	bool SavePNG(const FString& Filename, const TArray<FColor>& SrcBitmap, int32 Width, int32 Height);
};
