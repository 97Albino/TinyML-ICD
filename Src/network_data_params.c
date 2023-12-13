/**
  ******************************************************************************
  * @file    network_data_params.c
  * @author  AST Embedded Analytics Research Platform
  * @date    Fri Sep 30 01:56:55 2022
  * @brief   AI Tool Automatic Code Generator for Embedded NN computing
  ******************************************************************************
  * Copyright (c) 2022 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  ******************************************************************************
  */

#include "network_data_params.h"


/**  Activations Section  ****************************************************/
ai_handle g_network_activations_table[1 + 2] = {
  AI_HANDLE_PTR(AI_MAGIC_MARKER),
  AI_HANDLE_PTR(NULL),
  AI_HANDLE_PTR(AI_MAGIC_MARKER),
};




/**  Weights Section  ********************************************************/
AI_ALIGNED(32)
const ai_u64 s_network_weights_array_u64[284] = {
  0xbdb4d153bd94dbb5U, 0xbd814a6abdb1a092U, 0x3ca79668bcc6d618U, 0x3da772743d8ee435U,
  0x3d824bf93dbaaf07U, 0xbcd370473bfb4944U, 0xbd88f9dfbd76f66dU, 0xbd0778ecbd6dc639U,
  0xbe05290b3b47a40dU, 0xbe1799febe1e8e85U, 0xbd1046c0bdd52753U, 0x3e006ac53d2f3042U,
  0x3e1fa2183e11c745U, 0x3bcd21563dd70639U, 0xbddd3f37bd509988U, 0xbdc8ab06bdf033c3U,
  0x3c3916aebd54dbeeU, 0xbda38cc6be1cc826U, 0x3e2ce20a3d74c631U, 0x3e8199133e6981caU,
  0x3d1f72693e2abd2eU, 0xbe2a98b1bd5809aeU, 0xbe5d4270be73772bU, 0xbd763837be07a010U,
  0x3e33cb623d9f26b2U, 0x3dbdfde03e525d70U, 0x3e0f26dc3e006ed6U, 0x3d840f803def32c7U,
  0xbdb18440ba6d220fU, 0xbe15365fbdf67ab1U, 0xbd24e742bdf02f4cU, 0x3d9c443d3c5dc4ffU,
  0x3dc8e5653dc2adacU, 0x3ca3cf4d3d92bb51U, 0x3dcc2a463db9ba44U, 0x3d48b01e3daca4aaU,
  0xbd3d67b53b341669U, 0xbdb5fdf0bdb6469bU, 0xbd4badcfbdb77a0eU, 0x3d453c9f3c7d9756U,
  0x3d9a95b83d9b2a4dU, 0x3c8be0743d57de1dU, 0xbcbff3deU, 0x0U,
  0x0U, 0xbff413fcbf8ff681U, 0x3feed25f3f9bb24eU, 0x3dc074a03f865965U,
  0xc00b7dec3dc94614U, 0x3df02944becf8000U, 0xbf064929be8a7261U, 0x3e36d481bffa928fU,
  0x3f0b77633ef2d609U, 0xbffbce4c3f621782U, 0xbebeb666bf88d9ddU, 0x3fbecaba3f52f354U,
  0xbf7c308640129ce1U, 0x3414f1a1bf8a9b99U, 0x328b636e3480e465U, 0xb41c96f5b461aae8U,
  0xaffb3260af823e84U, 0x2efce133b114d312U, 0x3ce681112ffc688fU, 0xbb9eb734bd1bfe21U,
  0xbb695f443a27a581U, 0xb63286463764ccd9U, 0x3e89ef643dd0d1bdU, 0xbc3fba4a3e3ce3f3U,
  0xbea8406dbdfc8c5aU, 0xbee93b10bec62b1aU, 0xbe9c4870bef0c58fU, 0xbdbc3af6be2b5ce5U,
  0xbe0e951dbe290372U, 0x3d81ff69bb82b68eU, 0x3e99a2a03e59f266U, 0x3e76868b3e86e0fdU,
  0x3df89cd63e4199bcU, 0x3e3a7b5d3ce73abaU, 0xbd9f48fa3e139af7U, 0xbe27f28fbcea9dd2U,
  0xbe87a4c9be618a02U, 0xbe3bc90cbe99f4d0U, 0x3d246746bdfa821eU, 0x3db2f9433e3ba83bU,
  0xbd8ecac7bc24eb2cU, 0xbe9ecaacbe6f9a08U, 0xbeb2d71abea65fabU, 0xbde24d65be5ea67bU,
  0x3e721aa53e05f8caU, 0x3ca9e3823e388349U, 0xbe92e383bd9e1a39U, 0xbefa5ae0beaf59d1U,
  0xbea51d22bee5e823U, 0x36a8af74be17861cU, 0xb7650bb5b4ff1377U, 0x37882a7f3768d69dU,
  0x380dd12136f6b725U, 0x380046423806acd5U, 0xb600c11337cf72a0U, 0x35b97e7835bdc5a8U,
  0xb523e6c8b4af7db3U, 0xb57d02e7b5f55f28U, 0xb5967123b5b78da3U, 0xb6037bd8b606669aU,
  0xb5a4ff4cU, 0x0U, 0x0U, 0x0U,
  0xbf65e41c3f6c861bU, 0x3f6ba5613f51015eU, 0xb7dbe71b3f764056U, 0xad3dd1d63685b2baU,
  0xb06116b52edcd24cU, 0xaf7a87f8af442087U, 0x2c70f693adcfe835U, 0xab11f7ecabce9714U,
  0x2b0669932b702748U, 0xab4d82652a47a4c9U, 0x300ae71cabc5c606U, 0x2f809fcfaf6a2104U,
  0x2f99f2c52f6917ecU, 0xac86607eaef0a2f6U, 0xa9cb33e429bcbca1U, 0x293147b429a57628U,
  0xa9ee589029205f1cU, 0xac5b12ff2a3d64cbU, 0xac8548282b70624eU, 0xaba95911ab70056aU,
  0xab0a7adb2cdddbc1U, 0xa841b8aca90c6963U, 0x2729cb8d28dbd41dU, 0x28b48663a8112bedU,
  0x3f027e64a8a39fc7U, 0xb4bd0924b7a15646U, 0xb4000d8535b72831U, 0xb31a2edc3430c963U,
  0xbe9cf3f33f5e06d5U, 0xbee843d5bf04cd3dU, 0xbe0183ae3ed5346eU, 0xbdd953dabda71b66U,
  0xbd9ea7cd3e8fe38aU, 0xb7161720be39b539U, 0xb6aa90c035a9e803U, 0x358b67cab697c746U,
  0xb718b2a3353eaf5bU, 0xb72e9e42b7262a7dU, 0xb70eff4cb6997626U, 0xb370c6e0b30e2126U,
  0x33cca5fd33c55ba7U, 0xb2bdd00ab3200726U, 0x34356a9934143ff2U, 0x33a67ad333ba5a5aU,
  0xb4e5b14634578b6aU, 0x356d323a34c00769U, 0x357e304835c052efU, 0x33bd498e34f21b33U,
  0x349ee46a350959f0U, 0xb516de9fb44fc3ecU, 0xb290adb1b282d84dU, 0xb2fac935b28047dcU,
  0xb299c6a6b21e31c4U, 0xb2891badb1304d8aU, 0xb28bbb6fb14951e9U, 0x3235caf4b2178fd0U,
  0x33cb0a7b3374ffd2U, 0xb1017f0a337c9c28U, 0x337210a830b45774U, 0x338fe78033d2d525U,
  0x336675703345caf2U, 0xb109912fb21b8f4dU, 0x321caf9d2e9ce822U, 0x31490fa7b24cbe70U,
  0x316a49bab1c2f13aU, 0x318e3358b1f17963U, 0xb1af616dU, 0x0U,
  0x0U, 0x0U, 0x28501345299412aeU, 0x28b61a232982312eU,
  0x29483e6528975aaaU, 0xc05144bea95157e6U, 0xb68869bc39b9b04fU, 0xb5bf6b0438986948U,
  0xb5101d40b6c04ba6U, 0x3136173734b53096U, 0x2f33b3c03048f970U, 0xaef151822ea38542U,
  0xb4592b82afb9788dU, 0x2fda7f04326a1af2U, 0xae9bfbfc30f70c2cU, 0xae6d46dbaf0e4e7eU,
  0x27bb391baa45e6e6U, 0x28633457275fa0c8U, 0xa8ece03da80912faU, 0xaa412cf2a8f74d84U,
  0x288f4c44295309d0U, 0xa99abeb2a98d0869U, 0xa6b00a3e298c0b53U, 0x28ea1221aa288f25U,
  0x268e01c4a8c7f18eU, 0xa86bc792a8e384a5U, 0xb10c3f71a8f6e94eU, 0x35163da4beb934c6U,
  0xb2555f95b8225a1cU, 0xb281165eb0921e1dU, 0xb156c7e9b1cd92f2U, 0xb1a22780b1bc5cc9U,
  0x31b550b731a79569U, 0x31a75789319b3de4U, 0xb0c41d2731a9486dU, 0x3dba03a230ca4cddU,
  0x3e208ebf3dd13552U, 0x3e2a5cc23e3662cfU, 0x3e2751ce3e4f1f8cU, 0x3e40bfec3e6c140cU,
  0x3de2c5e23e5643e2U, 0x373ebf933746d584U, 0x3737eb3e37511524U, 0x375b24263770f4d8U,
  0x3729c7b637678d56U, 0x3716d62c3751bae9U, 0x37186cb3374896eeU, 0x37090ad636f12dadU,
  0x37276da73740387fU, 0x3724c75e3772dc7dU, 0x36c626153770647aU, 0x368df10537755556U,
  0xb1663071b17928d7U, 0x315f2cfc30abb285U, 0xb1a260b0307c933eU, 0x30c7361330331508U,
  0xb1b7e57631a89174U, 0x3000ae8b30c10332U, 0x31db1aef317c0bfaU, 0x320fe1cd310ad72dU,
  0x31c481883256295cU, 0x31840377318652ebU, 0xae31d0cd31e99926U, 0xb1e063d0314283d4U,
  0xb1d23d48b168dda9U, 0x31899dde2f30c091U, 0x31b03fefb1a8c446U, 0x310d0247b0db287eU,
  0x311e9990U, 0x0U, 0x0U, 0x0U,
  0xbefe4534b39f1a22U, 0x37d7464eb768ebb2U, 0x33895c43329adb42U, 0x33c9ef673326feeeU,
  0x3a86206440945458U, 0xb4633421ba5d1be0U, 0x340167c534ae2ec3U, 0xc0894372b4889e30U,
  0x3a42ab3fba66d426U, 0xb50de6e7b43eb078U, 0x3375d02bb4b5cee6U, 0xb680439bbe75c68cU,
  0x334e056f37067872U, 0x3319a3a5b3444af5U, 0x3f2d30c63374f369U, 0xb86c2aac37c1fb45U,
  0xb44db8e7b3062a62U, 0x328ff82133f1ee63U, 0x3a6cbc55408b1192U, 0x32ad42eaba446048U,
  0x3338c0aa34874477U, 0xbe5fe940325c45deU, 0x36dc5c92b661417aU, 0xb31d81ce2fadfbbcU,
  0x3df26855b2d14c28U, 0x3f83a1fabf8e40c6U, 0xbe2566af3d68aed4U, 0x3d53b54fbf855d7bU,
  0x3f824195bde6c469U, 0xbd676d27bf7127f3U, 0x3f74506b3e1bae76U, 0x3de6c469bd543fe2U,
  0x3f7127f3bf824195U, 0xbe1bae753d676d2aU, 0x3d543fe7bf74506bU, 0xbd8c94ab3d8c94d2U,
};


ai_handle g_network_weights_table[1 + 2] = {
  AI_HANDLE_PTR(AI_MAGIC_MARKER),
  AI_HANDLE_PTR(s_network_weights_array_u64),
  AI_HANDLE_PTR(AI_MAGIC_MARKER),
};

