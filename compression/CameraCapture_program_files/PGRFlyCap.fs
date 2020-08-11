module PGRFlyCap

open FlyCapture2Managed
open System.Runtime.InteropServices
open System.Xml.Linq
open System

let ig = ignore

let xn = XName.op_Implicit

let StartCam (videoAgent:MailboxProcessor<_>) width height (x:XElement) =
    let f s = x.Attribute(xn s) |> uint32
    let g s = x.Attribute(xn s) |> float32
    let sn = f "SerialNumber"
    let busmgr = new ManagedBusManager()
    let cam = new ManagedCamera()
    cam.Connect(busmgr.GetCameraFromSerialNumber(sn))
    let settings = Format7ImageSettings()
    settings.mode <- Mode.Mode0
    settings.width <- f "Width"
    settings.height <- f "Height"
    settings.pixelFormat <- PixelFormat.PixelFormat422Yuv8
    let info = cam.GetEmbeddedImageInfo()
    info.frameCounter.onOff <- true
    info.timestamp.onOff <- true
    cam.SetEmbeddedImageInfo(info)
    cam.SetFormat7Configuration(settings,100.0f)
    let fps = g "FPS"
    cam.SetProperty(CameraProperty(PropertyType.FrameRate,absValue = fps,autoManualMode=false,absControl=true,onOff=true))
    cam.SetProperty(CameraProperty(PropertyType.Shutter,absValue = g "Shutter",autoManualMode=false,absControl=true,onOff=true))
    cam.SetProperty(CameraProperty(PropertyType.Gain,absValue = g "Gain",autoManualMode=false,absControl=true,onOff=true))
    cam.SetGPIOPinDirection(2u,1u)
    //Unset Strobe Signal
    cam.SetStrobe(StrobeControl(source=2u,onOff=false,polarity=1u))
    Async.Sleep 1000 |> Async.RunSynchronously
    //Set Strobe Signal
    cam.SetStrobe(StrobeControl(source=2u,onOff=true,polarity=1u,duration=1000.0f/(2.0f*fps)))
    let x = new ManagedImage()
    let fint = Array.zeroCreate<byte> (width*height*2)
    let rec getData () = async {
        cam.RetrieveBuffer(x)
        Marshal.Copy(x.data|>NativeInterop.NativePtr.toNativeInt,fint,0,fint.Length)
        let metadata = 
            Array.concat
                [|BitConverter.GetBytes(x.imageMetadata.embeddedFrameCounter);BitConverter.GetBytes(x.imageMetadata.embeddedTimeStamp)|]
        videoAgent.Post(fint,metadata)
        return! getData ()   
    }
    cam.StartCapture()
    getData () |> Async.Start
    cam :> obj

let ColorSpace = "uyvy422"
