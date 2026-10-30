package com.voicestudio.mobile

import android.app.Application
import android.os.Looper
import androidx.compose.ui.test.*
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.lifecycle.viewModelScope
import androidx.test.core.app.ApplicationProvider
import com.voicestudio.mobile.data.*
import com.voicestudio.mobile.features.StudioViewModel
import kotlinx.coroutines.*
import org.junit.*
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.Shadows.shadowOf
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(sdk=[34])
class FirstUseFlowsTest {
    @get:Rule val compose=createComposeRule()

    private suspend fun withPhone(check: (StudioViewModel)->Unit) {
        val app=ApplicationProvider.getApplicationContext<Application>()
        androidx.work.testing.WorkManagerTestInitHelper.initializeTestWorkManager(app)
        shadowOf(app.getSystemService(android.app.ActivityManager::class.java)).setMemoryInfo(
            android.app.ActivityManager.MemoryInfo().apply {totalMem=16_000_000_000})
        val store=SessionStore(app,object:CredentialVault {
            override fun read(host:String):Secret?=null
            override fun write(host:String,secret:Secret?)=Unit
        })
        store.update {Preferences(consent=false)}
        val vm=StudioViewModel(app,store)
        try {
            compose.setContent {StudioApp(vm)}
            compose.waitUntil(15000) {shadowOf(Looper.getMainLooper()).idle();!vm.state.value.loading&&!vm.phone.state.value.loading}
            check(vm)
        } finally {vm.viewModelScope.cancel()}
    }

    @Test fun destinationsHaveVisibleTextAndHomeStartsCloningDirectly()=runBlocking {
        withPhone {vm ->
            for(label in listOf("Home","Voices","Dub","Dictate","Settings")) {
                compose.onAllNodes(hasText(label) and !hasClickAction(),useUnmergedTree=true).onLast().assertIsDisplayed()
            }
            compose.onNodeWithTag("home.voices").performScrollTo().performClick()
            compose.onNodeWithText("Voice name").assertExists()
            compose.onNodeWithText("Cancel").performScrollTo().performClick()
            compose.onNodeWithTag("home.speak").assertExists()
            Assert.assertFalse(vm.phone.downloads.progress.value.values.any {it.running})
        }
    }

    @Test fun speechSetupRetainsTheDraftAndRequiresExplicitLicenceConsent()=runBlocking {
        withPhone {vm ->
            compose.onNodeWithTag("home.speak").performScrollTo().performClick()
            compose.onNodeWithText("Type what you want to hear…").performTextInput("My first speech draft.")
            compose.onNodeWithText("Generate").assertDoesNotExist()
            compose.onNodeWithText("Download").assertIsEnabled().performClick()
            compose.onNodeWithText("Accept").assertIsNotEnabled()
            Assert.assertTrue(vm.phone.downloads.progress.value.isEmpty())
            compose.onNodeWithText("Decline").performClick()
            compose.onNodeWithText("My first speech draft.").assertExists()
            Assert.assertTrue(vm.phone.downloads.progress.value.isEmpty())
        }
    }

    @Test fun dictationExplainsTheRequiredDownload()=runBlocking {
        withPhone {
            compose.onNode(hasText("Dictate") and hasClickAction()).performClick()
            compose.onNodeWithText("to use this phone offline.",substring=true).assertExists()
            compose.onNodeWithText("Tap to start").assertDoesNotExist()
            compose.onNodeWithText("Download").assertIsEnabled()
        }
    }
}
