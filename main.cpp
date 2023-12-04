
#include "common.hpp"
// #include "cv.h"
// #include "highgui.h"
//#include "main.h"
#include "opencv2/opencv.hpp"
#define DLLEXPORT extern "C" //���� #include "stdafx.h" ֮��


unsigned char* g_pRgbData = NULL;
#define MAX_IMAGE_COUNT 10
//RGB8PlannerתBGR8Packed
bool ConvertRGB8Planner2BGR8Packed(const unsigned char* pSrcData,
                                   int nWidth,
                                   int nHeight,
                                   unsigned char* pDstData)
{
    if (NULL == pSrcData || NULL == pDstData)
    {
        return false;
    }
    int nImageStep = nWidth * nHeight;
    for (int i = 0; i < nImageStep; ++i)
    {
        pDstData[i * 3 + 0] = pSrcData[i + nImageStep * 2];
        pDstData[i * 3 + 1] = pSrcData[i + nImageStep * 1];
        pDstData[i * 3 + 2] = pSrcData[i + nImageStep * 0];
    }

    return true;
}
void* handle = NULL;

std::vector<MV3D_RGBD_DEVICE_INFO> devs(1);

DLLEXPORT void MV3D_RGBD_StartCapture() {
    LOGD("Initialize");
    ASSERT_OK(MV3D_RGBD_Initialize());
    unsigned int nDevNum = 0;
    ASSERT_OK(MV3D_RGBD_GetDeviceNumber(DeviceType_Ethernet | DeviceType_USB, &nDevNum));
    LOGD("MV3D_RGBD_GetDeviceNumber success! nDevNum:%d.", nDevNum);
    ASSERT(nDevNum);
    ASSERT_OK(MV3D_RGBD_GetDeviceList(DeviceType_Ethernet | DeviceType_USB, &devs[0], nDevNum, &nDevNum));
    ASSERT_OK(MV3D_RGBD_OpenDevice(&handle, &devs[0]));
    LOGD("OpenDevice success.");

    ASSERT_OK(MV3D_RGBD_Start(handle));
    LOGD("Start work success.");
}

DLLEXPORT void MV3D_RGBD_GetFrame() {
    BOOL bExit_Main = FALSE;
    MV3D_RGBD_FRAME_DATA stFrameData = { 0 };


    int nRet = MV3D_RGBD_FetchFrame(handle, &stFrameData, 5000);
    if (MV3D_RGBD_OK == nRet)
    {
        for (int i = 0; i < stFrameData.nImageCount; i++)
        {
            LOGD("MV3D_RGBD_FetchFrame Success: framenum (%d)(%d) height(%d) width(%d)  len (%d)!", i, stFrameData.stImageData[i].nFrameNum,
                 stFrameData.stImageData[i].nHeight, stFrameData.stImageData[i].nWidth, stFrameData.stImageData[i].nDataLen);


            if (ImageType_Depth == stFrameData.stImageData[i].enImageType)
            {
                cv::Mat  mCvmat = cv::Mat(stFrameData.stImageData[i].nHeight, stFrameData.stImageData[i].nWidth, CV_16UC1, stFrameData.stImageData[i].pData);
                char chFileName[256] = { 0 };
                // sprintf(chFileName, "Depth.png", stFrameData.stImageData[i].nFrameNum);
                cv::imwrite("Depth.png", mCvmat);
                // cvSaveImage(chFileName, &(IplImage(mCvmat)));

            }

            if (ImageType_RGB8_Planar == stFrameData.stImageData[i].enImageType)
            {
                if (NULL == g_pRgbData)
                {
                    g_pRgbData = (unsigned char*)malloc(stFrameData.stImageData[i].nDataLen);
                    if (NULL == g_pRgbData)
                    {
                        LOGD("MV3D_RGBD_FetchFrame: g_pRgbData malloc failed!");
                        bExit_Main = TRUE;
                        continue;
                    }
                    memset(g_pRgbData, 0, stFrameData.stImageData[i].nDataLen);
                }
                ConvertRGB8Planner2BGR8Packed(stFrameData.stImageData[i].pData, stFrameData.stImageData[i].nWidth, stFrameData.stImageData[i].nHeight, g_pRgbData);
                cv::Mat  mCvmat = cv::Mat(stFrameData.stImageData[i].nHeight, stFrameData.stImageData[i].nWidth, CV_8UC3, g_pRgbData);
                char chFileName[256] = { 0 };
                // sprintf(chFileName, "RGB.png", stFrameData.stImageData[i].nFrameNum);
                cv::imwrite("RGB.png", mCvmat);
                // cvSaveImage(chFileName, &(IplImage(mCvmat)));
            }
        }
    }
}

DLLEXPORT void MV3D_RGBD_EndCapture() {
    ASSERT_OK(MV3D_RGBD_Stop(handle));
    ASSERT_OK(MV3D_RGBD_CloseDevice(&handle));
    ASSERT_OK(MV3D_RGBD_Release());

    LOGD("Main done!");
}

int main(int argc,char** argv)
{
    LOGD("Initialize");
    ASSERT_OK( MV3D_RGBD_Initialize() );

    MV3D_RGBD_VERSION_INFO stVersion;
    ASSERT_OK( MV3D_RGBD_GetSDKVersion(&stVersion) );
    LOGD("dll version: %d.%d.%d", stVersion.nMajor, stVersion.nMinor, stVersion.nRevision);

    unsigned int nDevNum = 0;
    ASSERT_OK(MV3D_RGBD_GetDeviceNumber(DeviceType_Ethernet|DeviceType_USB, &nDevNum));
    LOGD("MV3D_RGBD_GetDeviceNumber success! nDevNum:%d.", nDevNum);
    ASSERT(nDevNum);

    // �����豸
    LOG("---------------------------------------------------------------\r\n");
    std::vector<MV3D_RGBD_DEVICE_INFO> devs(nDevNum);
    ASSERT_OK(MV3D_RGBD_GetDeviceList(DeviceType_Ethernet|DeviceType_USB, &devs[0], nDevNum, &nDevNum));

    for (unsigned int i = 0; i < nDevNum; i++)
    {
        if (DeviceType_Ethernet == devs[i].enDeviceType)
        {
            LOG("Index[%d]. SerialNum[%s] IP[%s] Name[%s].\r\n", i, devs[i].chSerialNumber, devs[i].SpecialInfo.stNetInfo.chCurrentIp, devs[i].chModelName);
        }
        else if (DeviceType_USB == devs[i].enDeviceType)
        {
            LOG("Index[%d]. SerialNum[%s] UsbProtocol[%d] Name[%s].\r\n", i, devs[i].chSerialNumber, devs[i].SpecialInfo.stUsbInfo.enUsbProtocol, devs[i].chModelName);
        }
    }

    LOG("---------------------------------------------------------------");

    unsigned int nIndex  = 0;
    while (true)
    {
        LOG("Please enter the index of the camera to be connected��\n");
        scanf("%d",&nIndex);
        LOG("Connected camera index:%d \r\n", nIndex);

        if ((nDevNum  <= nIndex) || (0 > nIndex))
        {
            LOG("enter error!\r\n");
        }
        else
        {
            break;
        }
    }
    LOG("---------------------------------------------------------------\r\n");

    void* handle = NULL;
    ASSERT_OK(MV3D_RGBD_OpenDevice(&handle, &devs[nIndex]));
    LOGD("OpenDevice success.");

    ASSERT_OK(MV3D_RGBD_Start(handle));
    LOGD("Start work success.");

    BOOL bExit_Main = FALSE;
    MV3D_RGBD_FRAME_DATA stFrameData = {0};
    int nDepthImgSaveCount = 0;
    int nRGBDImgSaveCount = 0;
    while (!bExit_Main )
    {
        // ��ȡͼ������
        int nRet = MV3D_RGBD_FetchFrame(handle, &stFrameData, 5000);
        if (MV3D_RGBD_OK == nRet)
        {
            for(int i = 0; i < stFrameData.nImageCount; i++)
            {
                LOGD("MV3D_RGBD_FetchFrame Success: framenum (%d)(%d) height(%d) width(%d)  len (%d)!", i,stFrameData.stImageData[i].nFrameNum,
                     stFrameData.stImageData[i].nHeight, stFrameData.stImageData[i].nWidth, stFrameData.stImageData[i].nDataLen);


                if (ImageType_Depth == stFrameData.stImageData[i].enImageType)
                {
                    cv::Mat  mCvmat = cv::Mat( stFrameData.stImageData[i].nHeight , stFrameData.stImageData[i].nWidth, CV_16UC1, stFrameData.stImageData[i].pData);
                    char chFileName[256] = {0};
                    sprintf(chFileName, "Depth_nFrameNum[%d].png",stFrameData.stImageData[i].nFrameNum);

                    if (MAX_IMAGE_COUNT > nDepthImgSaveCount)
                    {
                        // cvSaveImage(chFileName, &(IplImage(mCvmat)));
                        cv::imwrite(chFileName, mCvmat);
                        nDepthImgSaveCount++;
                    }
                }

                if (ImageType_RGB8_Planar == stFrameData.stImageData[i].enImageType)
                {
                    if(NULL == g_pRgbData)
                    {
                        g_pRgbData =  (unsigned char*)malloc(stFrameData.stImageData[i].nDataLen);
                        if (NULL == g_pRgbData)
                        {
                            LOGD("MV3D_RGBD_FetchFrame: g_pRgbData malloc failed!");
                            bExit_Main = TRUE;
                            continue;
                        }
                        memset(g_pRgbData, 0, stFrameData.stImageData[i].nDataLen);
                    }
                    ConvertRGB8Planner2BGR8Packed(stFrameData.stImageData[i].pData,stFrameData.stImageData[i].nWidth,stFrameData.stImageData[i].nHeight,g_pRgbData);
                    cv::Mat  mCvmat = cv::Mat( stFrameData.stImageData[i].nHeight , stFrameData.stImageData[i].nWidth, CV_8UC3, g_pRgbData);
                    char chFileName[256] = {0};
                    sprintf(chFileName, "RGB_nFrameNum[%d].png",stFrameData.stImageData[i].nFrameNum);

                    if (MAX_IMAGE_COUNT > nRGBDImgSaveCount)
                    {
                        // cvSaveImage(chFileName, &(IplImage(mCvmat)));
                        cv::imwrite(chFileName, mCvmat);
                        nRGBDImgSaveCount++;
                    }
                }
            }
        }
    }

    ASSERT_OK(MV3D_RGBD_Stop(handle));
    ASSERT_OK(MV3D_RGBD_CloseDevice(&handle));
    ASSERT_OK(MV3D_RGBD_Release());

    LOGD("Main done!");

    return  0;
}

