// Copyright (c) 2025 Florian Gutbier
// 
// This source code is part of the UE5 Plugin developed for the Bachelor's thesis
// at the University of Bamberg.
// 
// Released under the MIT License. See LICENSE file for details.


#include "UDatasetCaptureManager.h"
#include "Kismet/GameplayStatics.h"
#include "IImageWrapperModule.h"
#include "IImageWrapper.h"
#include "Components/LightComponent.h"
#include "Components/StaticMeshComponent.h"
#include "Components/LocalFogVolumeComponent.h"
#include "Engine/LocalFogVolume.h"
#include "Kismet/KismetMathLibrary.h"
#include "Engine/Light.h"
#include "Engine/Texture.h"
#include "Engine/Texture2D.h"
#include "Engine/TextureRenderTarget2D.h"
#include "Components/SceneCaptureComponent2D.h"
#include "UnrealClient.h"
#include "SceneView.h"
#include "Engine/LocalPlayer.h"
#include "RenderingThread.h"

// Define static variables for persisting state across level loads
FString UDatasetCaptureManager::s_DefaultBackgroundName = TEXT("");
bool UDatasetCaptureManager::s_bIsFirstCaptureEver = true;
bool UDatasetCaptureManager::s_bDummyScreenshotDone = false;

/***********************************************************************************************/
void UDatasetCaptureManager::Initialize(
    UWorld* World,
    ATargetPoint* ObjectTarget,
    const TArray<FVector>& CameraTargets,
    const TArray<FLinearColor>& LightColors,
    const TArray<UMaterialInterface*>& Materials,
    const TArray<TPair<TSoftObjectPtr<UStaticMesh>, int32>>& InMeshClassEntries,
    const TSoftObjectPtr<UWorld>& NextLevel,
    bool bAddFog,
    float InTargetRadius,
    UMaterialInterface* InMaskMaterial)
{
    if (!ensureAlwaysMsgf(World && ObjectTarget,
        TEXT("DatasetCaptureManager::Initialize called with null world or target")))
    {
        return;
    }

    // Initialize members
    m_pWorld = World;
    m_pObjectTarget = ObjectTarget;
    m_pNextLevel = NextLevel;
    m_vCameraTransforms = CameraTargets;
    m_aMaterials = Materials;
    m_aLightColors = LightColors;
    m_bAddFog = bAddFog;
    m_fTargetRadius = InTargetRadius;

    UE_LOG(LogTemp, Log, TEXT("Capture Manager: Initialize() called with InTargetRadius = %.2f cm"), InTargetRadius);
    UE_LOG(LogTemp, Log, TEXT("Capture Manager: m_fTargetRadius set to = %.2f cm"), m_fTargetRadius);

    m_sCurrentScreenshotPath = "";
    m_sRelativeImagePath = "";
    m_iCurrentMeshIndex = 0;
    m_iCurrentCameraIndex = 0;
    m_iCurrentLightColorIndex = 0;
    m_iCurrentMaterialIndex = 0;
    m_bFogActive = false;
    m_fCurrentObjectRadius = 0.f;
    m_vCurrentObjectCenter = FVector::ZeroVector;
    m_pCurrentSpawnedActor = nullptr;
    m_pMaskMaterial = InMaskMaterial;
    m_sCurrentMaskPath = "";
    m_sRelativeMaskPath = "";
    m_eCapturePhase = ECapturePhase::CameraVariations;

    // Store the default background name (the first level we're initialized in)
    // Use static variable to persist across level loads
    if (s_DefaultBackgroundName.IsEmpty())
    {
        s_DefaultBackgroundName = World ? World->GetMapName() : TEXT("DefaultLevel");
        UE_LOG(LogTemp, Log, TEXT("Capture Manager: Default background set to: %s"), *s_DefaultBackgroundName);
    }

    // Also store in instance variable for convenience
    m_sDefaultBackgroundName = s_DefaultBackgroundName;

    // Metadatawriter must be initialized before screenshot folder
    m_pMetadataWriter = NewObject<UDatasetMetadataWriter>();
    check(m_pMetadataWriter);
    m_pMetadataWriter->Initialize();

    SetupMeshEntries(InMeshClassEntries);
    CreateScreenshotFolder();

    // Capture original light colors from the map BEFORE setting up fog or anything else
    CaptureOriginalLightColors();

    if (m_bAddFog)
    {
        SetUpFog();
    }

    if (m_pMaskMaterial)
    {
        SetupMaskCapture();
    }
}

FString UDatasetCaptureManager::GetDatasetRoot() const
{
    return FPaths::ProjectDir() / TEXT("Dataset");
}

/***********************************************************************************************/
void UDatasetCaptureManager::SetupMeshEntries(
    const TArray<TPair<TSoftObjectPtr<UStaticMesh>, int32>>& InMeshClassEntries)
{
    m_aMeshEntries = InMeshClassEntries;
    m_iCurrentMeshIndex = 0;
}

/*----------------------------------------------------------------------------*/
void UDatasetCaptureManager::CreateScreenshotFolder() const
{
    IPlatformFile& PlatformFile = FPlatformFileManager::Get().GetPlatformFile();
    const FString  DatasetRoot = GetDatasetRoot();

    if (!PlatformFile.DirectoryExists(*DatasetRoot))
    {
        PlatformFile.CreateDirectoryTree(*DatasetRoot);
    }

    m_pMetadataWriter->setFilePath(FPaths::Combine(DatasetRoot, TEXT("Metadata.csv")));
}

/***********************************************************************************************/
void UDatasetCaptureManager::SetUpFog()
{
    if (!m_bAddFog || !m_pWorld || !m_pObjectTarget) return;

    // layout-related constants
    constexpr float  FogZOffset = 100.f;                    // +1 m up
    constexpr float  SphereRadius = 400.f;                    // 8 m diameter
    const FVector    SpawnLoc = m_pObjectTarget->GetActorLocation() +
        FVector(0.f, 0.f, FogZOffset);

    // visual look
    constexpr float        RadialDensity = 2.0f;
    constexpr float        HeightDensity = 2.0f;
    constexpr float        PhaseG = 0.15f;               // forward scatter
    const FLinearColor     Albedo = { 0.75f, 0.75f, 0.85f };

    m_pLocalFog = CreateTempFogVolume(
        SpawnLoc,
        FVector(SphereRadius),
        RadialDensity,
        HeightDensity,
        Albedo,
        PhaseG);

    if (m_pLocalFog)
    {
        m_pLocalFog->SetActorHiddenInGame(true);   //stay hidden until needed
    }
}

/***********************************************************************************************/
void UDatasetCaptureManager::StartCapture()
{
    if (!ensureMsgf(m_pMetadataWriter, TEXT("MetadataWriter is null"))) return;

    UE_LOG(LogTemp, Log, TEXT("Capture Manager: StartCapture() called - Current m_fTargetRadius = %.2f cm"), m_fTargetRadius);

    m_pMetadataWriter->CreateFile();
    m_pMetadataWriter->setLevelName(m_pWorld ? m_pWorld->GetMapName() : TEXT("Unknown"));

    // Check if current level is NOT the default background
    const FString CurrentMapName = m_pWorld ? m_pWorld->GetMapName() : TEXT("");
    const bool bIsBackgroundVariation = (CurrentMapName != m_sDefaultBackgroundName);

    if (bIsBackgroundVariation)
    {
        // In a background variation - we only capture with default settings (one image per model)
        UE_LOG(LogTemp, Log, TEXT("Capture Manager: Background variation detected (%s). Capturing with default settings only."), *CurrentMapName);
        // Set phase to BackgroundVariations so it uses the right folder structure
        m_eCapturePhase = ECapturePhase::BackgroundVariations;

        // Wait for level streaming to complete before processing captures
        // This ensures all textures and materials are fully loaded
        WaitForLevelStreaming();
        return;
    }

    // In default background - wait for level streaming before starting captures
    UE_LOG(LogTemp, Log, TEXT("Capture Manager: Default background detected. Waiting for level streaming..."));
    WaitForLevelStreaming();
}

// ============================================================================
// Poll until the level's streaming textures are fully loaded AND shaders compiled
// ============================================================================
void UDatasetCaptureManager::WaitForLevelStreaming()
{
    static constexpr float PollInterval = 0.2f; // seconds

    const int32 NumStillStreaming = IStreamingManager::Get().StreamAllResources(0.0f);

    if (NumStillStreaming == 0)
    {
        if (m_iWaitIterationCount == 0)
        {
            // First time streaming completes - log and start shader compilation wait
            UE_LOG(LogTemp, Log, TEXT("Capture Manager: Textures streamed in. Starting shader compilation wait (will wait up to 3 seconds)..."));
            FlushRenderingCommands();
            m_iWaitIterationCount = 1;

            // Increase wait time significantly for shader compilation (3 seconds in ~200ms intervals)
            // This ensures shaders have time to compile on the GPU
            FTimerHandle Handle;
            m_pWorld->GetTimerManager().SetTimer(
                Handle, this, &UDatasetCaptureManager::WaitForLevelStreaming,
                PollInterval, false);
            return;
        }
        else if (m_iWaitIterationCount < 15)  // Wait up to ~3 seconds (15 * 0.2s) for standard shader compilation
        {
            // Continue polling/flushing during shader compilation phase
            FlushRenderingCommands();
            ++m_iWaitIterationCount;

            if (m_iWaitIterationCount % 5 == 0)  // Log every 1 second instead of every 0.2s
            {
                UE_LOG(LogTemp, Log, TEXT("Capture Manager: Shader compilation phase %d/15 - flushing render commands..."), m_iWaitIterationCount);
            }

            FTimerHandle Handle;
            m_pWorld->GetTimerManager().SetTimer(
                Handle, this, &UDatasetCaptureManager::WaitForLevelStreaming,
                PollInterval, false);
            return;
        }
        else
        {
            // Shader compilation wait complete - final flush and proceed
            UE_LOG(LogTemp, Log, TEXT("Capture Manager: Shader compilation wait complete (waited %.1f seconds with %d flushes). Proceeding to capture..."), m_iWaitIterationCount * 0.2f, m_iWaitIterationCount);
            FlushRenderingCommands();
            m_iWaitIterationCount = 0;  // Reset for next level load
            ProcessCaptureState();
            return;
        }
    }
    else
    {
        // Still streaming textures
        m_iWaitIterationCount = 0;  // Reset if we need to re-stream
        UE_LOG(LogTemp, Log, TEXT("Capture Manager: Waiting for textures to stream... (%d resources pending)"), NumStillStreaming);
        FTimerHandle Handle;
        m_pWorld->GetTimerManager().SetTimer(
            Handle, this, &UDatasetCaptureManager::WaitForLevelStreaming,
            PollInterval, false);
    }
}

// ============================================================================
// Capture state machine
// 
// CAPTURE FLOW:
// 
// DEFAULT BACKGROUND (Level 1):
//   For each mesh:
//     Phase 1: Camera variations (angles 0-N with default light, material, no fog)
//     Phase 2: Light variations (reference camera with lights 0-N, default material, no fog)
//     Phase 3: Material variations (all cameras with materials 0-N, default light, no fog)
//     Phase 4: Fog variation (reference camera with fog, default light, material)
//   Output: Dataset/[ClassIndex]/Camera/Angle0-N/
//           Dataset/[ClassIndex]/Light/Light0-N/
//           Dataset/[ClassIndex]/Material/Mat0-N/
//           Dataset/[ClassIndex]/Fog/WithFog/
//
// BACKGROUND VARIATIONS (Level 2, Level 3, ...):
//   For each mesh: Capture ONE image with default settings
//   Output: Dataset/[ClassIndex]/Background/[LevelName]/
//
// Then proceed to next level if NextLevel is set, otherwise finish
// ============================================================================
void UDatasetCaptureManager::ProcessCaptureState()
{
    // ---- Guard clauses ----
    if (!m_pWorld)
    {
        UE_LOG(LogTemp, Warning, TEXT("Capture Manager: World is invalid."));
        FinalizeCapture();
        return;
    }

    if (m_iCurrentMeshIndex >= m_aMeshEntries.Num())
    {
        // All meshes processed in current background
        const FString CurrentMapName = m_pWorld->GetMapName();
        const bool bIsDefaultBackground = (CurrentMapName == m_sDefaultBackgroundName);

        if (bIsDefaultBackground && m_eCapturePhase != ECapturePhase::BackgroundVariations)
        {
            // In default background and haven't moved to background variations yet
            // Move to background variations phase
            m_eCapturePhase = ECapturePhase::BackgroundVariations;
            m_iCurrentMeshIndex = 0;  // Will be handled by BackgroundVariations phase
            ProcessCaptureState();
            return;
        }

        // Either we're in a background variation or we're done with background variations
        FinalizeCapture();
        return;
    }

    if (!m_pCurrentSpawnedActor)
    {
        SpawnCurrentMesh();
        return;
    }

    // ---- Local helpers ----
    auto SetDefaultState = [this]()
        {
            m_iCurrentCameraIndex = 0;
            m_iCurrentLightColorIndex = 0;
            m_iCurrentMaterialIndex = 0;
            if (m_pLocalFog) m_pLocalFog->SetActorHiddenInGame(true);
            m_bFogActive = false;
            m_pMetadataWriter->setIsFogEnabled(false);
            // Restore original map lighting instead of using configured light[0]
            RestoreOriginalLightColors();
        };

    // ---- Check if we're in a background variation ----
    const FString CurrentMapName = m_pWorld->GetMapName();
    const bool bIsDefaultBackground = (CurrentMapName == m_sDefaultBackgroundName);

    // If we're NOT in the default background, only capture one image with default settings
    if (!bIsDefaultBackground)
    {
        // Capture with default settings (camera 0, light 0, material default, no fog)
        CaptureScreenshotForCurrentCamera();

        // After screenshot is taken, OnScreenshotCaptured will call ProcessCaptureState again
        // We need to skip to the next mesh
        return;
    }

    // ---- Main state machine (only runs in DEFAULT background) ----

    // Phase 1: Camera variations (with default settings)
    if (m_eCapturePhase == ECapturePhase::CameraVariations)
    {
        if (m_iCurrentCameraIndex < m_aCameraTargets.Num())
        {
            CaptureScreenshotForCurrentCamera();
            return;
        }

        // Move to light variations
        m_eCapturePhase = ECapturePhase::LightVariations;
        m_iCurrentCameraIndex = 0;  // Use reference camera (first camera)
        m_iCurrentLightColorIndex = 0;  // Start from first light color
        ProcessCaptureState();
        return;
    }

    // Phase 2: Light color variations (with reference camera, default material, no fog)
    if (m_eCapturePhase == ECapturePhase::LightVariations)
    {
        if (m_iCurrentLightColorIndex < m_aLightColors.Num())
        {
            SetAllLightsColor(m_aLightColors[m_iCurrentLightColorIndex]);
            CaptureScreenshotForCurrentCamera();
            return;
        }

        // Move to material variations
        m_eCapturePhase = ECapturePhase::MaterialVariations;
        m_iCurrentCameraIndex = 0;  // Use reference camera
        m_iCurrentLightColorIndex = 0;  // Reset to default light
        // Restore original map lighting for material variations
        RestoreOriginalLightColors();
        m_iCurrentMaterialIndex = 0;  // Start from first custom material
        ProcessCaptureState();
        return;
    }

    // Phase 3: Material variations (with all cameras, default light, no fog)
    if (m_eCapturePhase == ECapturePhase::MaterialVariations)
    {
        if (m_aMaterials.IsValidIndex(m_iCurrentMaterialIndex))
        {
            if (m_iCurrentCameraIndex == 0)
            {
                ApplyCurrentMaterial();
            }

            if (m_iCurrentCameraIndex < m_aCameraTargets.Num())
            {
                CaptureScreenshotForCurrentCamera();
                return;
            }

            // Move to next material and capture its first camera
            ++m_iCurrentMaterialIndex;
            m_iCurrentCameraIndex = 0;
            ProcessCaptureState();
            return;
        }

        // Move to fog variation
        m_eCapturePhase = ECapturePhase::FogVariation;
        m_iCurrentCameraIndex = 0;  // Use reference camera
        m_iCurrentLightColorIndex = 0;  // Reset to default light
        // Restore original map lighting for fog variations
        RestoreOriginalLightColors();

        // Restore original materials and wait for render thread to update
        RestoreOriginalMaterials();

        // Add delay to allow materials to be applied before screenshot
        constexpr float MaterialRestoreDelay = 0.1f;
        FTimerHandle Handle;
        m_pWorld->GetTimerManager().SetTimer(Handle, [this]()
        {
            ProcessCaptureState();
        }, MaterialRestoreDelay, false);
        return;
    }

    // Phase 4: Fog variation (with reference camera, default light, default material)
    if (m_eCapturePhase == ECapturePhase::FogVariation)
    {
        if (m_bAddFog && !m_bFogActive)
        {
            if (m_pLocalFog) m_pLocalFog->SetActorHiddenInGame(false);
            m_bFogActive = true;
            m_pMetadataWriter->setIsFogEnabled(true);
            CaptureScreenshotForCurrentCamera();
            return;
        }

        // All factor variations complete for this mesh
        m_eCapturePhase = ECapturePhase::Complete;
        ProcessCaptureState();
        return;
    }

    // Phase 5: Background variations (load next background and capture all meshes)
    if (m_eCapturePhase == ECapturePhase::BackgroundVariations)
    {
        // If we have a next level and haven't loaded it yet, load it
        if (!m_pNextLevel.IsNull())
        {
            const FName LevelName(*m_pNextLevel.ToSoftObjectPath().GetLongPackageName());
            if (!LevelName.IsNone())
            {
                UE_LOG(LogTemp, Log, TEXT("Capture Manager: Loading next background: %s"), *LevelName.ToString());
                UGameplayStatics::OpenLevel(m_pWorld, LevelName);
                return;  // OpenLevel will destroy current world, execution stops here
            }
        }

        // No more levels to process
        m_eCapturePhase = ECapturePhase::Complete;
        ProcessCaptureState();
        return;
    }

    // Phase 6: Complete - clean up and move to next mesh
    if (m_eCapturePhase == ECapturePhase::Complete)
    {
        if (!m_pCurrentSpawnedActor->IsPendingKillPending())
        {
            m_pCurrentSpawnedActor->Destroy();
        }
        if (m_pLocalFog) m_pLocalFog->SetActorHiddenInGame(true);
        m_bFogActive = false;

        m_pCurrentSpawnedActor = nullptr;
        ++m_iCurrentMeshIndex;

        // Reset for next mesh
        m_eCapturePhase = bIsDefaultBackground ? ECapturePhase::CameraVariations : ECapturePhase::BackgroundVariations;
        SetDefaultState();

        ProcessCaptureState();
        return;
    }
}

// ============================================================================
// Mesh spawning
// ============================================================================
void UDatasetCaptureManager::SpawnCurrentMesh()
{
    const TPair<TSoftObjectPtr<UStaticMesh>, int32>& Entry = m_aMeshEntries[m_iCurrentMeshIndex];
    UStaticMesh* Mesh = Entry.Key.LoadSynchronous();

    if (!ensureAlwaysMsgf(Mesh, TEXT("Failed to load mesh at index %d"), m_iCurrentMeshIndex))
    {
        ++m_iCurrentMeshIndex;
        ProcessCaptureState();
        return;
    }

    const FTransform SpawnXf = m_pObjectTarget
        ? m_pObjectTarget->GetActorTransform()
        : FTransform::Identity;

    FActorSpawnParameters SpawnParams;
    SpawnParams.SpawnCollisionHandlingOverride = ESpawnActorCollisionHandlingMethod::AlwaysSpawn;

    m_pCurrentSpawnedActor = m_pWorld->SpawnActor<AActor>(AActor::StaticClass(), SpawnXf, SpawnParams);
    if (!ensureAlwaysMsgf(m_pCurrentSpawnedActor,
        TEXT("Failed to spawn actor for mesh %s"), *Mesh->GetName()))
    {
        ++m_iCurrentMeshIndex;
        ProcessCaptureState();
        return;
    }

    // Add a StaticMeshComponent and assign the mesh
    UStaticMeshComponent* MeshComp = NewObject<UStaticMeshComponent>(m_pCurrentSpawnedActor);
    MeshComp->SetStaticMesh(Mesh);
    MeshComp->RegisterComponent();
    m_pCurrentSpawnedActor->SetRootComponent(MeshComp);

    // Force this mesh's textures to stream at full resolution immediately
    Mesh->SetForceMipLevelsToBeResident(30.f);
    const TArray<FStaticMaterial>& StaticMaterials = Mesh->GetStaticMaterials();
    for (const FStaticMaterial& StaticMat : StaticMaterials)
    {
        if (UMaterialInterface* Mat = StaticMat.MaterialInterface)
        {
            Mat->SetForceMipLevelsToBeResident(false, true, 30.f);
        }
    }

    // Store target rotation before we modify transform
    const FRotator TargetRotation = SpawnXf.Rotator();

    // Normalize size so all meshes have a consistent bounding sphere radius
    UE_LOG(LogTemp, Log, TEXT("Capture Manager: About to call NormalizeActorScale() with m_fTargetRadius = %.2f cm"), m_fTargetRadius);
    NormalizeActorScale();

    // Ensure the actor is positioned correctly at the target point:
    // - XY axes: Center the model (normal behavior)
    // - Z axis: Use the lowest point of the model to prevent ground clipping
    // First get the current bounds after scaling
    const FBox Bounds = m_pCurrentSpawnedActor->GetComponentsBoundingBox(true);
    const FVector BoundsCenter = Bounds.GetCenter();
    const FVector BoundsMin = Bounds.Min;

    // Calculate offset to move actor:
    // X, Y: center alignment at target point
    // Z: lowest point alignment at target point (prevents clipping through ground)
    const FVector TargetLocation = SpawnXf.GetLocation();
    FVector OffsetToTarget;
    OffsetToTarget.X = TargetLocation.X - BoundsCenter.X;
    OffsetToTarget.Y = TargetLocation.Y - BoundsCenter.Y;
    OffsetToTarget.Z = TargetLocation.Z - BoundsMin.Z;  // Use lowest point for Z, not center

    m_pCurrentSpawnedActor->AddActorWorldOffset(OffsetToTarget);

    // Reapply the target rotation to ensure it's not lost during repositioning
    m_pCurrentSpawnedActor->SetActorRotation(TargetRotation);

    UE_LOG(LogTemp, Log, TEXT("Capture Manager: Positioned mesh at target point. Offset applied: (%.2f, %.2f, %.2f), Rotation: (%.1f, %.1f, %.1f)°"), 
        OffsetToTarget.X, OffsetToTarget.Y, OffsetToTarget.Z,
        TargetRotation.Pitch, TargetRotation.Yaw, TargetRotation.Roll);

    UE_LOG(LogTemp, Log, TEXT("Capture Manager: Spawned mesh %s"), *Mesh->GetName());

    BuildCameraTargetsForCurrentActor();

    // Store original materials for later restoration
    m_OriginalMaterials.Reset();
    TArray<UMeshComponent*> Meshes;
    m_pCurrentSpawnedActor->GetComponents<UMeshComponent>(Meshes);
    for (UMeshComponent* MeshComponent : Meshes)
    {
        TArray<UMaterialInterface*> OriginalMats;
        const int32 NumSlots = MeshComponent->GetNumMaterials();
        for (int32 Slot = 0; Slot < NumSlots; ++Slot)
        {
            OriginalMats.Add(MeshComponent->GetMaterial(Slot));
        }
        m_OriginalMaterials.Add(MeshComponent, OriginalMats);
    }

    // metadata
    m_pMetadataWriter->setModelName(Mesh->GetName());
    m_pMetadataWriter->setMaterialName(TEXT("Default"));

    // Ensure correct lights and metadata for first pass - use original map lighting
    m_iCurrentCameraIndex = 0;
    m_iCurrentLightColorIndex = 0;
    RestoreOriginalLightColors();

    ProcessCaptureState();
}

// ============================================================================
// Normalize the spawned actor's scale so its bounding sphere matches the target radius
// ============================================================================
void UDatasetCaptureManager::NormalizeActorScale()
{
    if (!m_pCurrentSpawnedActor || m_fTargetRadius <= 0.f)
    {
        if (m_fTargetRadius <= 0.f)
        {
            UE_LOG(LogTemp, Warning, TEXT("Capture Manager: NormalizeActorScale() called but m_fTargetRadius = %.2f (invalid)"), m_fTargetRadius);
        }
        return;
    }

    const FBox Bounds = m_pCurrentSpawnedActor->GetComponentsBoundingBox(true);
    const FVector Extents = Bounds.GetExtent();

    // Use the LARGEST extent dimension for consistent sizing across different aspect ratios
    // This ensures objects with different shapes (thin planes, cubic bins, etc.)
    // appear the same screen size when scaled to the same target radius.
    // 
    // Previously used Bounds.GetExtent().Size() which is the 3D diagonal magnitude.
    // This caused thin objects (e.g., airplanes) to scale differently than cubic objects.
    const float LargestExtent = FMath::Max3(Extents.X, Extents.Y, Extents.Z);

    if (LargestExtent < KINDA_SMALL_NUMBER)
    {
        UE_LOG(LogTemp, Warning, TEXT("Capture Manager: NormalizeActorScale() - mesh size too small: %.2f"), LargestExtent);
        return;
    }

    const float ScaleFactor = m_fTargetRadius / LargestExtent;
    m_pCurrentSpawnedActor->SetActorScale3D(FVector(ScaleFactor));

    UE_LOG(LogTemp, Log, TEXT("Capture Manager: NormalizeActorScale() - Target: %.2f cm, Largest Extent: %.2f cm (X=%.2f, Y=%.2f, Z=%.2f), ScaleFactor: %.4f"), 
        m_fTargetRadius, LargestExtent, Extents.X, Extents.Y, Extents.Z, ScaleFactor);
}

// ============================================================================
// Screenshot capture
// ============================================================================
void UDatasetCaptureManager::CaptureScreenshotForCurrentCamera()
{
    static constexpr float ScreenshotDelay = 0.3f; // seconds - increased for material loading

    const FVector& CamLoc = m_aCameraTargets[m_iCurrentCameraIndex];

    if (APlayerController* PC = m_pWorld->GetFirstPlayerController())
    {
        if (APawn* Pawn = PC->GetPawn())
        {
            const FRotator LookAt =
                UKismetMathLibrary::FindLookAtRotation(CamLoc, m_vCurrentObjectCenter);

            Pawn->SetActorLocation(CamLoc);
            PC->SetControlRotation(LookAt);
        }
    }

    // Reposition sky sphere to camera location (only needed for first capture, but harmless for others)
    // This ensures the sky material fully covers the viewport
    TArray<AActor*> SkySpheres;
    UGameplayStatics::GetAllActorsOfClass(m_pWorld, AActor::StaticClass(), SkySpheres);
    for (AActor* Actor : SkySpheres)
    {
        FString ActorName = Actor->GetName();
        // Check for common sky sphere naming patterns
        if (ActorName.Contains(TEXT("Sky")) || ActorName.Contains(TEXT("Atmosphere")) || ActorName.Contains(TEXT("BP_Sky")))
        {
            // Move sky sphere to camera location so it fully envelops the view
            Actor->SetActorLocation(CamLoc);
            UE_LOG(LogTemp, Log, TEXT("Capture Manager: Repositioned sky actor (%s) to camera location"), *ActorName);
            break;  // Only reposition the first sky sphere found
        }
    }

    m_pMetadataWriter->setCameraPosition(m_vCameraTransforms[m_iCurrentCameraIndex].ToString());

    /* ------------------------------------------------------------------ */
    const int32   ClassIndex = m_aMeshEntries[m_iCurrentMeshIndex].Value;
    const FString MapName = m_pWorld->GetMapName();
    const FString MeshName = m_pCurrentSpawnedActor->GetName();
    const FString DatasetRoot = GetDatasetRoot();

    // Check if we're in the default background or a background variation
    const bool bIsDefaultBackground = (MapName == m_sDefaultBackgroundName);

    // Determine folder structure based on current phase and background
    FString FactorFolder;
    FString VariationFolder;

    if (!bIsDefaultBackground)
    {
        // We're in a background variation - always use Background folder
        FactorFolder = TEXT("Background");
        VariationFolder = MapName;
    }
    else
    {
        // We're in the default background - use factor-based structure
        switch (m_eCapturePhase)
        {
            case ECapturePhase::CameraVariations:
                FactorFolder = TEXT("Camera");
                VariationFolder = FString::Printf(TEXT("Angle%d"), m_iCurrentCameraIndex);
                break;

            case ECapturePhase::LightVariations:
                FactorFolder = TEXT("Light");
                VariationFolder = FString::Printf(TEXT("Light%d"), m_iCurrentLightColorIndex);
                break;

            case ECapturePhase::MaterialVariations:
                FactorFolder = TEXT("Material");
                if (m_aMaterials.IsValidIndex(m_iCurrentMaterialIndex))
                {
                    VariationFolder = FPaths::Combine(m_aMaterials[m_iCurrentMaterialIndex]->GetName(), FString::Printf(TEXT("Angle%d"), m_iCurrentCameraIndex));
                }
                else
                {
                    VariationFolder = FPaths::Combine(TEXT("DefaultMaterial"), FString::Printf(TEXT("Angle%d"), m_iCurrentCameraIndex));
                }
                break;

            case ECapturePhase::FogVariation:
                FactorFolder = TEXT("Fog");
                VariationFolder = TEXT("WithFog");
                break;

            default:
                FactorFolder = TEXT("Unknown");
                VariationFolder = TEXT("Unknown");
                break;
        }
    }

    // Build path: Dataset/FactorFolder/VariationFolder/
    const FString DatasetRootRel = FPaths::Combine(DatasetRoot, FactorFolder, VariationFolder);
    const FString DatasetRootAbs = FPaths::Combine(FPaths::ProjectDir(), DatasetRootRel);

    IPlatformFile& PF = FPlatformFileManager::Get().GetPlatformFile();
    PF.CreateDirectoryTree(*DatasetRootAbs);

    // Image naming: MapName_MeshName_ClassIndex.png
    const auto MakeScreenshotName = [&](const FString& Dir) -> FString
        {
            return FString::Printf(
                TEXT("%s/%s_%s_%d.png"),
                *Dir,
                *MapName,
                *MeshName,
                ClassIndex);
        };

    m_sCurrentScreenshotPath = MakeScreenshotName(DatasetRootAbs);
    m_sRelativeImagePath = MakeScreenshotName(DatasetRootRel);

    // Build mask paths (same name with _mask suffix) - if enabled
    if (m_pMaskMaterial)
    {
        const auto MakeMaskName = [&](const FString& Dir) -> FString
            {
                return FString::Printf(
                    TEXT("%s/%s_%s_%d_mask.png"),
                    *Dir,
                    *MapName,
                    *MeshName,
                    ClassIndex);
            };
        m_sCurrentMaskPath = MakeMaskName(DatasetRootAbs);
        m_sRelativeMaskPath = MakeMaskName(DatasetRootRel);
    }

    m_pMetadataWriter->setClassIndex(FString::FromInt(ClassIndex));

    // Fire the screenshot after a delay to ensure the next frame was rendered.
    float ActualScreenshotDelay = ScreenshotDelay;

    // Add extra delay for the very first capture in the level to allow Eye Adaptation (auto-exposure) to settle
    bool bIsFirstCaptureInLevel = false;
    if (bIsDefaultBackground)
    {
        bIsFirstCaptureInLevel = (m_iCurrentMeshIndex == 0 && m_iCurrentCameraIndex == 0 && m_eCapturePhase == ECapturePhase::CameraVariations);
    }
    else
    {
        bIsFirstCaptureInLevel = (m_iCurrentMeshIndex == 0); // 1 picture per mesh in background variations
    }

    if (bIsFirstCaptureInLevel)
    {
        ActualScreenshotDelay = 3.5f; // 3.5 seconds to ensure exposure has fully settled
        UE_LOG(LogTemp, Log, TEXT("Capture Manager: Extra delay (%.1fs) for first capture to let eye adaptation settle."), ActualScreenshotDelay);
    }

    FTimerHandle Handle;
    m_pWorld->GetTimerManager()
        .SetTimer(Handle, this,
            &UDatasetCaptureManager::RequestCameraScreenshot,
            ActualScreenshotDelay, false);
}

// ============================================================================
// Request a screenshot of the current camera
// ============================================================================
void UDatasetCaptureManager::RequestCameraScreenshot()
{
    UGameViewportClient* Viewport = m_pWorld ? m_pWorld->GetGameViewport() : nullptr;
    if (!Viewport)
    {
        UE_LOG(LogTemp, Warning, TEXT("Capture Manager: No GameViewportClient found."));
        ++m_iCurrentCameraIndex;
        ProcessCaptureState();
        return;
    }

    // Ensure only one delegate
    if (m_oScreenshotCapturedHandle.IsValid())
    {
        Viewport->OnScreenshotCaptured().Remove(m_oScreenshotCapturedHandle);
        m_oScreenshotCapturedHandle.Reset();
    }

    m_oScreenshotCapturedHandle = Viewport->OnScreenshotCaptured()
        .AddUObject(this, &UDatasetCaptureManager::OnScreenshotCaptured);

    // Add a small delay before requesting the screenshot to ensure GPU is ready
    constexpr float PreScreenshotDelay = 0.15f;
    FTimerHandle PreScreenshotHandle;
    m_pWorld->GetTimerManager().SetTimer(
        PreScreenshotHandle, 
        [this]()
        {
            FScreenshotRequest::RequestScreenshot(m_sCurrentScreenshotPath, /*bShowUI=*/false, /*bAddFilenameSuffix=*/false);
        },
        PreScreenshotDelay, 
        false);
}

// ============================================================================
// Helper to spawn a temporary local-fog volume
// ============================================================================
ALocalFogVolume* UDatasetCaptureManager::CreateTempFogVolume(
    const FVector&       Location,
    const FVector&       UniformScale,
    float                RadialDensity,
    float                HeightDensity,
    const FLinearColor&  Albedo,
    float                PhaseG)
{
    if (!m_pWorld) return nullptr;

    FActorSpawnParameters Params;
    Params.SpawnCollisionHandlingOverride = ESpawnActorCollisionHandlingMethod::AlwaysSpawn;

    ALocalFogVolume* FogActor =
        m_pWorld->SpawnActor<ALocalFogVolume>(Location, FRotator::ZeroRotator, Params);
    if (!FogActor) return nullptr;

    // convert cm -> m (could also just use meters)
    constexpr float UnitsPerMetre = 100.f;
    FogActor->SetActorScale3D(UniformScale / UnitsPerMetre);

    if (ULocalFogVolumeComponent* Fog = FogActor->FindComponentByClass<ULocalFogVolumeComponent>())
    {
        Fog->SetRadialFogExtinction(RadialDensity);
        Fog->SetHeightFogExtinction(HeightDensity);
        Fog->SetFogAlbedo(Albedo);
        Fog->SetFogPhaseG(PhaseG);
        Fog->SetVisibility(true);
        Fog->RegisterComponent();
    }
    return FogActor;
}


// ============================================================================
// Apply a colour to every ALight in the level
// ============================================================================
void UDatasetCaptureManager::SetAllLightsColor(const FLinearColor& NewColor) const
{
    if (!m_pWorld) return;

    TArray<AActor*> Lights;
    UGameplayStatics::GetAllActorsOfClass(m_pWorld, ALight::StaticClass(), Lights);

    for (AActor* Actor : Lights)
    {
        if (ALight* Light = Cast<ALight>(Actor))
        {
            if (ULightComponent* LC = Light->GetLightComponent())
            {
                LC->SetLightColor(NewColor);
            }
        }
    }

    if (m_pMetadataWriter)
    {
        m_pMetadataWriter->setLightColor(NewColor.ToString());
    }
}

// ============================================================================
// Capture the original light colors from all lights in the world
// ============================================================================
void UDatasetCaptureManager::CaptureOriginalLightColors()
{
    if (!m_pWorld)
    {
        UE_LOG(LogTemp, Warning, TEXT("Capture Manager: Cannot capture light colors - world is null"));
        return;
    }

    m_OriginalLightColors.Reset();

    TArray<AActor*> Lights;
    UGameplayStatics::GetAllActorsOfClass(m_pWorld, ALight::StaticClass(), Lights);

    for (AActor* Actor : Lights)
    {
        if (ALight* Light = Cast<ALight>(Actor))
        {
            if (ULightComponent* LC = Light->GetLightComponent())
            {
                m_OriginalLightColors.Add(Actor, LC->GetLightColor());
            }
        }
    }

    UE_LOG(LogTemp, Log, TEXT("Capture Manager: Captured original colors for %d lights"), m_OriginalLightColors.Num());
}

// ============================================================================
// Restore all lights to their original colors
// ============================================================================
void UDatasetCaptureManager::RestoreOriginalLightColors() const
{
    if (!m_pWorld || m_OriginalLightColors.Num() == 0)
    {
        return;
    }

    for (const auto& Pair : m_OriginalLightColors)
    {
        AActor* Actor = Pair.Key;
        const FLinearColor& OriginalColor = Pair.Value;

        if (Actor && Actor->IsValidLowLevel())
        {
            if (ALight* Light = Cast<ALight>(Actor))
            {
                if (ULightComponent* LC = Light->GetLightComponent())
                {
                    LC->SetLightColor(OriginalColor);
                }
            }
        }
    }
}

// ============================================================================
// Apply a material to every mesh slot on the current actor
// ============================================================================
void UDatasetCaptureManager::ApplyCurrentMaterial()
{
    if (!m_pCurrentSpawnedActor ||
        !m_aMaterials.IsValidIndex(m_iCurrentMaterialIndex))
    {
        return;
    }

    UMaterialInterface* Mat = m_aMaterials[m_iCurrentMaterialIndex];
    if (!Mat) return;

    TArray<UMeshComponent*> Meshes;
    m_pCurrentSpawnedActor->GetComponents<UMeshComponent>(Meshes);

    for (UMeshComponent* Mesh : Meshes)
    {
        const int32 NumSlots = Mesh->GetNumMaterials();
        for (int32 Slot = 0; Slot < NumSlots; ++Slot)
        {
            Mesh->SetMaterial(Slot, Mat);
        }
    }

    if (m_pMetadataWriter)
    {
        m_pMetadataWriter->setMaterialName(Mat->GetName());
    }
}

// ============================================================================
// Restore original materials on the current actor
// ============================================================================
void UDatasetCaptureManager::RestoreOriginalMaterials()
{
    if (!m_pCurrentSpawnedActor)
    {
        UE_LOG(LogTemp, Warning, TEXT("Capture Manager: Cannot restore materials - no spawned actor"));
        return;
    }

    UE_LOG(LogTemp, Log, TEXT("Capture Manager: Restoring original materials..."));

    int32 RestoredMeshCount = 0;
    for (const TPair<UMeshComponent*, TArray<UMaterialInterface*>>& Pair : m_OriginalMaterials)
    {
        UMeshComponent* Mesh = Pair.Key;
        const TArray<UMaterialInterface*>& OriginalMats = Pair.Value;

        if (!Mesh || !Mesh->IsValidLowLevel())
        {
            UE_LOG(LogTemp, Warning, TEXT("Capture Manager: Invalid mesh component during restore"));
            continue;
        }

        for (int32 Slot = 0; Slot < OriginalMats.Num(); ++Slot)
        {
            Mesh->SetMaterial(Slot, OriginalMats[Slot]);
        }
        RestoredMeshCount++;
    }

    if (m_pMetadataWriter)
    {
        m_pMetadataWriter->setMaterialName(TEXT("Default"));
    }

    UE_LOG(LogTemp, Log, TEXT("Capture Manager: Restored original materials on %d mesh component(s)"), RestoredMeshCount);
}

// ============================================================================
// Screenshot callback
// ============================================================================
void UDatasetCaptureManager::OnScreenshotCaptured(
    int32            Width,
    int32            Height,
    const TArray<FColor>& Bitmap)
{
    /* Always unhook the delegate first */
    if (UGameViewportClient* VP = m_pWorld ? m_pWorld->GetGameViewport() : nullptr)
    {
        VP->OnScreenshotCaptured().Remove(m_oScreenshotCapturedHandle);
    }
    m_oScreenshotCapturedHandle.Reset();

    /* Dummy first screenshot for GPU warmup - don't save */
    const FString CurrentMapName = m_pWorld ? m_pWorld->GetMapName() : TEXT("");
    const bool bIsDefaultBackground = (CurrentMapName == m_sDefaultBackgroundName);

    if (bIsDefaultBackground && s_bIsFirstCaptureEver && !s_bDummyScreenshotDone)
    {
        s_bDummyScreenshotDone = true;
        UE_LOG(LogTemp, Log, TEXT("Capture Manager: Dummy first screenshot captured. Warming up mask capture system..."));

        // Dummy mask capture to warm up the mask system if enabled
        if (m_pMaskMaterial && m_pMaskCaptureComponent && m_pCurrentSpawnedActor)
        {
            // Set up mask capture component the same way as real capture
            const FVector& CamLoc = m_aCameraTargets[m_iCurrentCameraIndex];
            const FRotator LookAt = UKismetMathLibrary::FindLookAtRotation(CamLoc, m_vCurrentObjectCenter);
            m_pMaskCaptureComponent->SetWorldLocationAndRotation(CamLoc, LookAt);

            // Apply mask material temporarily for warmup
            TArray<UStaticMeshComponent*> MeshComps;
            m_pCurrentSpawnedActor->GetComponents<UStaticMeshComponent>(MeshComps);
            TArray<TArray<UMaterialInterface*>> OriginalMats;
            OriginalMats.SetNum(MeshComps.Num());

            for (int32 c = 0; c < MeshComps.Num(); ++c)
            {
                UStaticMeshComponent* Comp = MeshComps[c];
                if (!Comp) continue;
                const int32 NumSlots = Comp->GetNumMaterials();
                OriginalMats[c].SetNum(NumSlots);
                for (int32 s = 0; s < NumSlots; ++s)
                {
                    OriginalMats[c][s] = Comp->GetMaterial(s);
                    Comp->SetMaterial(s, m_pMaskMaterial);
                }
            }

            FlushRenderingCommands();

            m_pMaskCaptureComponent->ShowOnlyActors.Reset();
            m_pMaskCaptureComponent->ShowOnlyActors.Add(m_pCurrentSpawnedActor);
            m_pMaskCaptureComponent->CaptureScene();
            FlushRenderingCommands();

            // Restore original materials
            for (int32 c = 0; c < MeshComps.Num(); ++c)
            {
                UStaticMeshComponent* Comp = MeshComps[c];
                if (!Comp) continue;
                for (int32 s = 0; s < OriginalMats[c].Num(); ++s)
                {
                    Comp->SetMaterial(s, OriginalMats[c][s]);
                }
            }

            FlushRenderingCommands();
            UE_LOG(LogTemp, Log, TEXT("Capture Manager: Dummy mask capture completed for GPU warmup."));
        }

        // Add a delay to allow the GPU to complete rendering of the dummy screenshot
        constexpr float DummyWarmupDelay = 1.0f; // 1 second for GPU to fully complete the warmup render
        FTimerHandle DummyWarmupHandle;
        m_pWorld->GetTimerManager().SetTimer(
            DummyWarmupHandle, this, &UDatasetCaptureManager::ProcessCaptureState,
            DummyWarmupDelay, false);
        return;
    }

    /* Save PNG & write metadata */
    if (!m_sCurrentScreenshotPath.IsEmpty() &&
        Bitmap.Num() == Width * Height &&
        SavePNG(*m_sCurrentScreenshotPath, Bitmap, Width, Height))
    {
        UE_LOG(LogTemp, Log, TEXT("Capture Manager: Screenshot saved: %s"), *m_sCurrentScreenshotPath);

        // Capture segmentation mask if enabled
        if (m_pMaskMaterial)
        {
            CaptureSegmentationMask(Width, Height);
        }

        if (m_pMetadataWriter)
        {
            m_pMetadataWriter->setImageName(m_sRelativeImagePath);
            m_pMetadataWriter->setMaskImageName(
                m_pMaskMaterial ? m_sRelativeMaskPath : TEXT(""));
            m_pMetadataWriter->WriteToFile();
        }
    }
    else
    {
        UE_LOG(LogTemp, Warning, TEXT("Capture Manager: Screenshot save FAILED: %s"), *m_sCurrentScreenshotPath);
    }

    /* Advance the state machine */
    const bool bIsDefaultBackgroundForStateMachine = (CurrentMapName == m_sDefaultBackgroundName);

    if (!bIsDefaultBackgroundForStateMachine)
    {
        // In background variation - move to next mesh after each screenshot
        if (m_pCurrentSpawnedActor && !m_pCurrentSpawnedActor->IsPendingKillPending())
        {
            m_pCurrentSpawnedActor->Destroy();
        }
        m_pCurrentSpawnedActor = nullptr;
        ++m_iCurrentMeshIndex;
    }
    else
    {
        // In default background - increment appropriate index based on current phase
        if (m_eCapturePhase == ECapturePhase::CameraVariations || m_eCapturePhase == ECapturePhase::MaterialVariations)
        {
            ++m_iCurrentCameraIndex;
        }
        else if (m_eCapturePhase == ECapturePhase::LightVariations)
        {
            ++m_iCurrentLightColorIndex;
        }
    }

    ProcessCaptureState();
}

// ============================================================================
// Set up the SceneCaptureComponent2D and render target for segmentation masks
// ============================================================================
void UDatasetCaptureManager::SetupMaskCapture()
{
    if (!m_pWorld || !m_pMaskMaterial) return;

    // Get viewport resolution for render target size
    int32 Width = 512;
    int32 Height = 512;
    if (UGameViewportClient* Viewport = m_pWorld->GetGameViewport())
    {
        FVector2D ViewportSize;
        Viewport->GetViewportSize(ViewportSize);
        if (ViewportSize.X > 0 && ViewportSize.Y > 0)
        {
            Width = FMath::RoundToInt32(ViewportSize.X);
            Height = FMath::RoundToInt32(ViewportSize.Y);
        }
    }

    // Create render target with explicit format to ensure compatibility
    m_pMaskRenderTarget = NewObject<UTextureRenderTarget2D>(this);
    m_pMaskRenderTarget->RenderTargetFormat = ETextureRenderTargetFormat::RTF_RGBA8;
    m_pMaskRenderTarget->InitAutoFormat(Width, Height);
    m_pMaskRenderTarget->ClearColor = FLinearColor::Black;
    m_pMaskRenderTarget->UpdateResourceImmediate(true);

    // Create a persistent actor to hold the capture component
    FActorSpawnParameters SpawnParams;
    SpawnParams.SpawnCollisionHandlingOverride = ESpawnActorCollisionHandlingMethod::AlwaysSpawn;
    AActor* CaptureActor = m_pWorld->SpawnActor<AActor>(AActor::StaticClass(), FTransform::Identity, SpawnParams);
    if (!CaptureActor) 
    {
        UE_LOG(LogTemp, Warning, TEXT("Capture Manager: Failed to spawn capture actor"));
        return;
    }

    m_pMaskCaptureComponent = NewObject<USceneCaptureComponent2D>(CaptureActor);
    if (!m_pMaskCaptureComponent)
    {
        UE_LOG(LogTemp, Warning, TEXT("Capture Manager: Failed to create capture component"));
        return;
    }

    m_pMaskCaptureComponent->RegisterComponent();
    CaptureActor->SetRootComponent(m_pMaskCaptureComponent);

    m_pMaskCaptureComponent->TextureTarget = m_pMaskRenderTarget;
    // Use SCS_SceneColorHDR to get proper alpha channel for mask rendering
    m_pMaskCaptureComponent->CaptureSource = ESceneCaptureSource::SCS_SceneColorHDR;
    m_pMaskCaptureComponent->bCaptureEveryFrame = false;
    m_pMaskCaptureComponent->bCaptureOnMovement = false;

    // Only render actors explicitly added to ShowOnlyActors (black background for everything else)
    m_pMaskCaptureComponent->PrimitiveRenderMode = ESceneCapturePrimitiveRenderMode::PRM_UseShowOnlyList;

    // Ensure the component is enabled
    m_pMaskCaptureComponent->SetComponentTickEnabled(true);
    m_pMaskCaptureComponent->SetVisibility(true);
    m_pMaskCaptureComponent->SetHiddenInGame(false);

    // Store actor reference to prevent it from being garbage collected
    m_pCaptureActor = CaptureActor;

    UE_LOG(LogTemp, Log, TEXT("Capture Manager: Mask capture set up (%dx%d) with RTF_RGBA8 format."), Width, Height);
}

// ============================================================================
// Capture and save a segmentation mask for the current camera position
// ============================================================================
void UDatasetCaptureManager::CaptureSegmentationMask(int32 ScreenshotWidth, int32 ScreenshotHeight)
{
    if (!m_pMaskCaptureComponent || !m_pMaskRenderTarget || !m_pWorld || !m_pCurrentSpawnedActor) return;

    // Resize render target to match the actual screenshot resolution.
    // This ensures the aspect ratio is identical to the main camera's projection,
    // which prevents the object from appearing at a different size in the mask.
    if (ScreenshotWidth > 0 && ScreenshotHeight > 0 &&
        (m_pMaskRenderTarget->SizeX != ScreenshotWidth || m_pMaskRenderTarget->SizeY != ScreenshotHeight))
    {
        m_pMaskRenderTarget->InitAutoFormat(ScreenshotWidth, ScreenshotHeight);
        UE_LOG(LogTemp, Log, TEXT("Capture Manager: Mask RT resized to match screenshot: %dx%d"), ScreenshotWidth, ScreenshotHeight);
    }

    // Ensure the capture component only renders our spawned actor
    m_pMaskCaptureComponent->ShowOnlyActors.Reset();
    m_pMaskCaptureComponent->ShowOnlyActors.Add(m_pCurrentSpawnedActor);

    // Use the exact same camera position and look-at used for the main screenshot.
    const FVector& CamLoc = m_aCameraTargets[m_iCurrentCameraIndex];
    const FRotator LookAt = UKismetMathLibrary::FindLookAtRotation(CamLoc, m_vCurrentObjectCenter);
    m_pMaskCaptureComponent->SetWorldLocationAndRotation(CamLoc, LookAt);

    // Match the main camera's projection exactly by extracting its actual
    // projection matrix.  Simply copying GetFOVAngle() is insufficient because
    // UE5's AspectRatioAxisConstraint (default: MaintainYFOV) adjusts the
    // effective horizontal FOV based on viewport aspect ratio, while
    // SceneCaptureComponent2D uses FOVAngle as raw horizontal FOV.
    // Using the custom projection matrix bypasses this mismatch entirely.

    // SIMPLIFIED: Use basic FOV matching instead of complex projection matrix
    // This avoids near-plane culling issues that can occur with custom matrices
    m_pMaskCaptureComponent->bUseCustomProjectionMatrix = false;

    float MaskFOV = 50.f;  // Default to 50° FOV for consistency
    if (APlayerController* PC = m_pWorld->GetFirstPlayerController())
    {
        if (PC->PlayerCameraManager)
        {
            MaskFOV = PC->PlayerCameraManager->GetFOVAngle();
            UE_LOG(LogTemp, Log, TEXT("Capture Manager: Using player camera FOV: %.1f°"), MaskFOV);
        }
    }
    m_pMaskCaptureComponent->FOVAngle = MaskFOV;

    UE_LOG(LogTemp, Log, TEXT("Capture Manager: Mask capture FOV set to %.1f°"), MaskFOV);

    // Swap materials to the mask material
    TArray<UStaticMeshComponent*> MeshComps;
    m_pCurrentSpawnedActor->GetComponents<UStaticMeshComponent>(MeshComps);

    UE_LOG(LogTemp, Log, TEXT("Capture Manager: Found %d mesh components for mask capture"), MeshComps.Num());

    // Store original materials so we can restore them after capture
    TArray<TArray<UMaterialInterface*>> OriginalMaterials;
    OriginalMaterials.SetNum(MeshComps.Num());

    for (int32 c = 0; c < MeshComps.Num(); ++c)
    {
        UStaticMeshComponent* Comp = MeshComps[c];
        if (!Comp) continue;

        const int32 NumSlots = Comp->GetNumMaterials();
        OriginalMaterials[c].SetNum(NumSlots);
        for (int32 s = 0; s < NumSlots; ++s)
        {
            OriginalMaterials[c][s] = Comp->GetMaterial(s);
            Comp->SetMaterial(s, m_pMaskMaterial);
        }
        UE_LOG(LogTemp, Log, TEXT("Capture Manager: Applied mask material to component %d (%s) with %d slots"), 
            c, *Comp->GetName(), NumSlots);
    }

    // Verify mask material is valid
    if (!m_pMaskMaterial)
    {
        UE_LOG(LogTemp, Warning, TEXT("Capture Manager: Mask material is NULL!"));
    }
    else
    {
        UE_LOG(LogTemp, Log, TEXT("Capture Manager: Mask material: %s"), *m_pMaskMaterial->GetName());
    }

    // Flush rendering to ensure mask materials are applied before capture
    FlushRenderingCommands();

    // Debug: Log camera setup
    UE_LOG(LogTemp, Log, TEXT("Capture Manager: Mask capture camera at (%.0f, %.0f, %.0f) looking at (%.0f, %.0f, %.0f), FOV: %.1f°"), 
        CamLoc.X, CamLoc.Y, CamLoc.Z,
        m_vCurrentObjectCenter.X, m_vCurrentObjectCenter.Y, m_vCurrentObjectCenter.Z,
        m_pMaskCaptureComponent->FOVAngle);

    // Debug: Log object bounds
    const FBox ObjectBounds = m_pCurrentSpawnedActor->GetComponentsBoundingBox(true);
    UE_LOG(LogTemp, Log, TEXT("Capture Manager: Object bounds - Center: (%.0f, %.0f, %.0f), Size: (%.0f, %.0f, %.0f)"), 
        ObjectBounds.GetCenter().X, ObjectBounds.GetCenter().Y, ObjectBounds.GetCenter().Z,
        ObjectBounds.GetSize().X, ObjectBounds.GetSize().Y, ObjectBounds.GetSize().Z);

    // CRITICAL FIX: Force rebuild of ShowOnlyList by immediately recapturing with proper frustum
    // This ensures the render state is synchronized with the updated actor list
    m_pMaskCaptureComponent->CaptureScene();
    FlushRenderingCommands();

    // Log that capture was initiated
    UE_LOG(LogTemp, Log, TEXT("Capture Manager: Initial mask capture triggered (to sync render state)"));

    // Hide fog and atmospherics during mask capture — volumetric effects bypass ShowOnlyActors
    const bool bFogWasVisible = m_pLocalFog && !m_pLocalFog->IsHidden();
    if (bFogWasVisible)
    {
        m_pLocalFog->SetActorHiddenInGame(true);
    }

    TArray<AActor*> HiddenAtmospherics;
    TArray<AActor*> AllActors;
    UGameplayStatics::GetAllActorsOfClass(m_pWorld, AActor::StaticClass(), AllActors);

    for (AActor* Actor : AllActors)
    {
        if (!Actor || Actor == m_pCurrentSpawnedActor || Actor == m_pLocalFog || Actor->IsHidden()) continue;

        FString ClassName = Actor->GetClass()->GetName();
        FString ActorName = Actor->GetName();

        if (ClassName.Contains(TEXT("Fog")) || ClassName.Contains(TEXT("Cloud")) || 
            ClassName.Contains(TEXT("Sky")) || ClassName.Contains(TEXT("Atmosphere")) ||
            ActorName.Contains(TEXT("Fog")) || ActorName.Contains(TEXT("Cloud")) || 
            ActorName.Contains(TEXT("Sky")) || ActorName.Contains(TEXT("Atmosphere")))
        {
            Actor->SetActorHiddenInGame(true);
            HiddenAtmospherics.Add(Actor);
        }
    }

    // Capture the scene
    m_pMaskCaptureComponent->CaptureScene();

    // Flush the rendering thread to ensure the capture completes before we read pixels
    FlushRenderingCommands();

    // Restore fog visibility
    if (bFogWasVisible)
    {
        m_pLocalFog->SetActorHiddenInGame(false);
    }

    // Restore atmospheric actors visibility
    for (AActor* Actor : HiddenAtmospherics)
    {
        if (Actor)
        {
            Actor->SetActorHiddenInGame(false);
        }
    }

    // Read back pixels from the render target (MUST be after FlushRenderingCommands)
    TArray<FColor> MaskPixels;
    FTextureRenderTargetResource* RTResource = m_pMaskRenderTarget->GameThread_GetRenderTargetResource();
    if (!RTResource)
    {
        UE_LOG(LogTemp, Warning, TEXT("Capture Manager: Mask render target resource is null."));
        // Restore original materials before returning
        for (int32 c = 0; c < MeshComps.Num(); ++c)
        {
            UStaticMeshComponent* Comp = MeshComps[c];
            if (!Comp) continue;

            for (int32 s = 0; s < OriginalMaterials[c].Num(); ++s)
            {
                Comp->SetMaterial(s, OriginalMaterials[c][s]);
            }
        }
        return;
    }

    FReadSurfaceDataFlags ReadFlags(RCM_UNorm);
    RTResource->ReadPixels(MaskPixels, ReadFlags);

    // Diagnostic logging for first capture
    if (MaskPixels.Num() > 0)
    {
        FColor FirstPixel = MaskPixels[0];
        FColor MidPixel = MaskPixels[MaskPixels.Num() / 2];
        uint32 WhiteCount = 0;
        uint32 BlackCount = 0;
        for (const FColor& Pixel : MaskPixels)
        {
            if (Pixel.R > 200 && Pixel.G > 200 && Pixel.B > 200)
                ++WhiteCount;
            else if (Pixel.R < 50 && Pixel.G < 50 && Pixel.B < 50)
                ++BlackCount;
        }
        UE_LOG(LogTemp, Log, TEXT("Capture Manager: Mask pixels read - Size: %d, First: RGBA(%d,%d,%d,%d), Mid: RGBA(%d,%d,%d,%d), White%%: %.1f%%, Black%%: %.1f%%"),
            MaskPixels.Num(),
            FirstPixel.R, FirstPixel.G, FirstPixel.B, FirstPixel.A,
            MidPixel.R, MidPixel.G, MidPixel.B, MidPixel.A,
            100.0f * WhiteCount / MaskPixels.Num(),
            100.0f * BlackCount / MaskPixels.Num());
    }

    // Restore original materials AFTER reading pixels
    for (int32 c = 0; c < MeshComps.Num(); ++c)
    {
        UStaticMeshComponent* Comp = MeshComps[c];
        if (!Comp) continue;

        for (int32 s = 0; s < OriginalMaterials[c].Num(); ++s)
        {
            Comp->SetMaterial(s, OriginalMaterials[c][s]);
        }
    }

    // With SCS_SceneColor and ShowOnlyList:
    // - Object rendered with mask material: looks at luminance and alpha from mask material
    // - Background (not in ShowOnlyList): render target clear color (black), alpha = 0
    // 
    // The mask material should output white (255,255,255) for the object to segment,
    // and the background will be black (0,0,0).
    // Create binary mask: white = object, black = background
    for (FColor& Pixel : MaskPixels)
    {
        // If pixel is bright (object rendered with white mask material)
        const uint32 Luminance = (uint32(Pixel.R) + uint32(Pixel.G) + uint32(Pixel.B)) / 3;
        const bool bIsObject = (Luminance > 127);  // Threshold at 50% brightness

        Pixel.R = bIsObject ? 255 : 0;
        Pixel.G = bIsObject ? 255 : 0;
        Pixel.B = bIsObject ? 255 : 0;
        Pixel.A = 255;
    }

    const int32 Width = m_pMaskRenderTarget->SizeX;
    const int32 Height = m_pMaskRenderTarget->SizeY;

    if (MaskPixels.Num() == Width * Height &&
        SavePNG(m_sCurrentMaskPath, MaskPixels, Width, Height))
    {
        UE_LOG(LogTemp, Log, TEXT("Capture Manager: Mask saved: %s"), *m_sCurrentMaskPath);
    }
    else
    {
        UE_LOG(LogTemp, Warning, TEXT("Capture Manager: Mask save FAILED: %s"), *m_sCurrentMaskPath);
    }
}

// ============================================================================
// Finish a full dataset pass
// ============================================================================
void UDatasetCaptureManager::FinalizeCapture()
{
    const FString CurrentMapName = m_pWorld ? m_pWorld->GetMapName() : TEXT("");
    const bool bIsDefaultBackground = (CurrentMapName == m_sDefaultBackgroundName);

    // Check if we should load the next level (works for both default and background variations)
    if (!m_pNextLevel.IsNull())
    {
        const FName LevelName(*m_pNextLevel.ToSoftObjectPath().GetLongPackageName());
        if (!LevelName.IsNone())
        {
            if (bIsDefaultBackground)
            {
                UE_LOG(LogTemp, Log, TEXT("Capture Manager: Default background complete. Loading next background: %s"), *LevelName.ToString());
            }
            else
            {
                UE_LOG(LogTemp, Log, TEXT("Capture Manager: Background variation (%s) complete. Loading next background: %s"), *CurrentMapName, *LevelName.ToString());
            }
            UGameplayStatics::OpenLevel(m_pWorld, LevelName);
            return;
        }
    }

    // No more levels to process
    if (bIsDefaultBackground)
    {
        UE_LOG(LogTemp, Log, TEXT("Capture Manager: All captures complete! No more backgrounds."));
    }
    else
    {
        UE_LOG(LogTemp, Log, TEXT("Capture Manager: Background variation (%s) complete! No more backgrounds."), *CurrentMapName);
    }

    RemoveFromRoot();
}

// ============================================================================
// Raw -> PNG helper
// ============================================================================
bool UDatasetCaptureManager::SavePNG(
    const FString& Filename,
    const TArray<FColor>& SrcBitmap,
    int32                 Width,
    int32                 Height)
{
    IImageWrapperModule& ImageWrapper =
        FModuleManager::LoadModuleChecked<IImageWrapperModule>("ImageWrapper");

    TSharedPtr<IImageWrapper> Png = ImageWrapper.CreateImageWrapper(EImageFormat::PNG);
    if (Png.IsValid() &&
        Png->SetRaw(
            SrcBitmap.GetData(),
            SrcBitmap.GetAllocatedSize(),
            Width, Height,
            ERGBFormat::BGRA, 8))
    {
        const TArray64<uint8>& Data = Png->GetCompressed(100);   // max quality
        return FFileHelper::SaveArrayToFile(Data, *Filename);
    }
    return false;
}

// ============================================================================
// Build camera locations that perfectly frame the current actor
// ============================================================================
void UDatasetCaptureManager::BuildCameraTargetsForCurrentActor()
{
    if (!m_pCurrentSpawnedActor || !m_pWorld) return;

    // Bounding sphere of the actor
    const FBox  Bounds = m_pCurrentSpawnedActor->GetComponentsBoundingBox(true);
    m_vCurrentObjectCenter = Bounds.GetCenter();
    m_fCurrentObjectRadius = Bounds.GetExtent().Size();

    // Current camera FoV and Viewport Aspect Ratio
    float CamFOVdeg = 50.f;
    float ViewportAspect = 16.0f / 9.0f; // Default assumption
    if (APlayerController* PC = m_pWorld->GetFirstPlayerController())
    {
        if (PC->PlayerCameraManager)
            CamFOVdeg = PC->PlayerCameraManager->GetFOVAngle();
    }

    if (UGameViewportClient* Viewport = m_pWorld->GetGameViewport())
    {
        FVector2D ViewportSize;
        Viewport->GetViewportSize(ViewportSize);
        if (ViewportSize.Y > 0)
            ViewportAspect = ViewportSize.X / ViewportSize.Y;
    }

    // Use dynamic camera distance based on projected bounds.
    // This perfectly normalizes the apparent visual size across different view angles.
    const FVector Extents = Bounds.GetExtent();

    m_aCameraTargets.Reset();

    const FTransform Ref =
        m_pObjectTarget ? m_pObjectTarget->GetActorTransform() : FTransform::Identity;

    for (const FVector& Dir : m_vCameraTransforms)
    {
        const FVector WorldDir = Ref.GetRotation().RotateVector(Dir.GetSafeNormal());

        // Determine orthogonal camera plane axes
        FVector CamRight = FVector::CrossProduct(WorldDir, FVector::UpVector);
        if (CamRight.SizeSquared() < 0.01f) CamRight = FVector::CrossProduct(WorldDir, FVector::ForwardVector);
        CamRight.Normalize();
        FVector CamUp = FVector::CrossProduct(CamRight, WorldDir).GetSafeNormal();

        float HalfFovRad = FMath::DegreesToRadians(CamFOVdeg * 0.5f);
        float TanHalfHoriz = FMath::Tan(HalfFovRad);

        float MaxRequiredDistance = 0.f;

        for (int i = 0; i < 8; ++i)
        {
            FVector Corner(
                (i & 1) ? Extents.X : -Extents.X,
                (i & 2) ? Extents.Y : -Extents.Y,
                (i & 4) ? Extents.Z : -Extents.Z
            );

            // Project corner onto camera axes
            float Right = FMath::Abs(FVector::DotProduct(Corner, CamRight));
            float Up    = FMath::Abs(FVector::DotProduct(Corner, CamUp));
            float Depth = FVector::DotProduct(Corner, WorldDir); // Distance towards camera from center

            // Required distance to fit horizontally and vertically for this corner
            float ReqDistRight = (Right / TanHalfHoriz) + Depth;
            float ReqDistUp    = (Up * ViewportAspect / TanHalfHoriz) + Depth;

            MaxRequiredDistance = FMath::Max3(MaxRequiredDistance, ReqDistRight, ReqDistUp);
        }

        // Map m_fTargetRadius to screen percentage coverage (e.g., 50 -> 50% fraction)
        float FractionOfScreen = FMath::Clamp(m_fTargetRadius / 100.f, 0.05f, 2.0f);
        float DynamicCamDistance = MaxRequiredDistance / FractionOfScreen;

        // Ensure we don't clip inside the mesh's physical bounding sphere
        DynamicCamDistance = FMath::Max(DynamicCamDistance, m_fCurrentObjectRadius * 1.5f + 10.f);

        m_aCameraTargets.Add(m_vCurrentObjectCenter + WorldDir * DynamicCamDistance);
    }

    m_iCurrentCameraIndex = 0;  // reset camera index
}

// ============================================================================
// Load mesh-class mapping from a CSV file
// ============================================================================
bool UDatasetCaptureManager::LoadMeshClassMapFromCSV(
    const FString& CSVFilePath,
    TArray<TPair<TSoftObjectPtr<UStaticMesh>, int32>>& OutEntries)
{
    OutEntries.Reset();

    TArray<FString> Lines;
    if (!FFileHelper::LoadFileToStringArray(Lines, *CSVFilePath))
    {
        UE_LOG(LogTemp, Warning, TEXT("LoadMeshClassMapFromCSV: Failed to read file: %s"), *CSVFilePath);
        return false;
    }

    for (int32 i = 0; i < Lines.Num(); ++i)
    {
        const FString& Line = Lines[i];
        if (Line.IsEmpty() || Line.StartsWith(TEXT("#")) || Line.StartsWith(TEXT("//")))
        {
            continue; // skip empty lines and comments
        }

        FString MeshPath, ClassStr;
        if (!Line.Split(TEXT(";"), &MeshPath, &ClassStr))
        {
            UE_LOG(LogTemp, Warning, TEXT("LoadMeshClassMapFromCSV: Skipping malformed line %d: %s"), i + 1, *Line);
            continue;
        }

        MeshPath.TrimStartAndEndInline();
        ClassStr.TrimStartAndEndInline();

        if (!FCString::IsNumeric(*ClassStr))
        {
            UE_LOG(LogTemp, Warning, TEXT("LoadMeshClassMapFromCSV: Non-numeric class ID on line %d: %s"), i + 1, *ClassStr);
            continue;
        }

        const int32 ClassIndex = FCString::Atoi(*ClassStr);
        TSoftObjectPtr<UStaticMesh> MeshPtr{FSoftObjectPath(MeshPath)};

        OutEntries.Add(TPair<TSoftObjectPtr<UStaticMesh>, int32>(MoveTemp(MeshPtr), ClassIndex));
    }

    UE_LOG(LogTemp, Log, TEXT("LoadMeshClassMapFromCSV: Loaded %d entries from %s"), OutEntries.Num(), *CSVFilePath);
    return OutEntries.Num() > 0;
}

