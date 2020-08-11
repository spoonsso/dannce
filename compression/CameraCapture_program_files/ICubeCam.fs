module ICubeCam
open System.Runtime.InteropServices

type callback = delegate of nativeint*int*int->unit

let [<Literal>] ICubeDLL = "ICubeSDK64.dll"
[<DllImport(ICubeDLL)>]
extern int ICubeSDK_Init();
[<DllImport(ICubeDLL)>]
extern int ICubeSDK_Open(int nCamIndex);
[<DllImport(ICubeDLL)>]
extern int ICubeSDK_Close(int nCamIndex);
[<DllImport(ICubeDLL)>]
extern int ICubeSDK_Stop(int nCamIndex);
[<DllImport(ICubeDLL)>]
extern int ICubeSDK_SetMode(int nCamIndex,int nMode);
[<DllImport(ICubeDLL)>]
extern int ICubeSDK_SetCamParameter(int nCamIndex,int Type,uint32 Value);
[<DllImport(ICubeDLL)>]
extern int ICubeSDK_SetTrigger(int nCamIndex,int nMode);
[<DllImport(ICubeDLL)>]
extern int ICubeSDK_Start(int nCamIndex,nativeint ImgHandle, int Preview, int Callback);
[<DllImport(ICubeDLL)>]
extern int ICubeSDK_SetCallback(int nCamIndex,int nMode, nativeint pCallbackFunc, int pCBContext);

open System.Xml.Linq
open System
open System.Windows.Media.Imaging
open System.Windows.Media
let ig = ignore

let xn = XName.op_Implicit

let StartCam (videoAgent:MailboxProcessor<_>) width height (x:XElement) =
    let f s = x.Attribute(xn s) |> int
    let devnum = f "DeviceNumber"
    let mode,exposure,gain,clockrate = f "Mode",f "Exposure",f "Gain",f "ClockRate"

    ICubeSDK_Init()|>ig // Initialize
    ICubeSDK_Open(devnum)|>ig // Open
    ICubeSDK_SetMode(devnum, mode)|>ig // Set Mode            
    ICubeSDK_SetCamParameter(devnum, 7, exposure|>uint32)|>ig // Set Exposure time (ms)
    ICubeSDK_SetCamParameter(devnum, 15, clockrate|>uint32)|>ig // Set clock rate (MHz)
    ICubeSDK_SetCamParameter(devnum, 13, gain|>uint32)|>ig // Set Gain
    ICubeSDK_SetCamParameter(devnum, 117, 0u)|>ig // Set Color Processing Off
    ICubeSDK_SetCamParameter(devnum, 111, 0u)|>ig // Disable gamma
    
    ICubeSDK_SetCamParameter(devnum, 21, 1u)|>ig // Trigger on rising edge
    ICubeSDK_SetTrigger(devnum, 2)|>ig //Trigger source is hardware
    
    let cb = callback(fun pBuf lBufferSize pCBContext ->
        let frame = Array.zeroCreate<byte> (width*height*3)
        Marshal.Copy(pBuf,frame,0,lBufferSize)
        videoAgent.Post(frame,[||])
    )
    ICubeSDK_SetCallback(devnum,1,Marshal.GetFunctionPointerForDelegate(cb),0)|>ig //1 = RGB, - = RAW
    ICubeSDK_Start(devnum,System.IntPtr.Zero,0,1)|>ig
    cb :> obj

let ColorSpace = "bgr8"
